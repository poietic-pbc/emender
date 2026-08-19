#!/usr/bin/env python3
"""Synchronous full-model masked SFT for the dense 513B-token E97 authority."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch, torch.distributed as dist
from schedulefree import AdamWScheduleFree
from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity, sha256
from ndm.e97 import load_e97_checkpoint

def emit(path,event,**kw):
 if dist.get_rank()==0:
  with path.open('a') as f: f.write(json.dumps({'event':event,'time_unix':time.time(),**kw},sort_keys=True)+'\n')
def sync_grads(model):
 for p in model.parameters():
  if p.grad is not None: dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
def objective(model,tokens,masks,length,spans,global_targets):
 total=tokens.new_zeros((),dtype=torch.float32); observed=0
 for start,stop in spans:
  real=stop-start
  if real<2: raise RuntimeError('record too short')
  padded=((real-2)//16+1)*16+1; x=torch.zeros((1,padded),device=tokens.device,dtype=torch.long); x[:,:real]=tokens[:,start:stop]
  mask=torch.zeros((1,padded-1),device=tokens.device,dtype=torch.bool); mask[:,:real-1]=masks[:,start+1:stop]; observed+=int(mask.sum())
  part=model(x,return_loss=True,actual_length=torch.tensor([real],device=x.device),loss_mask=mask,loss_reduction='sum')
  (part/global_targets).backward(); total+=part.detach().float()
 if observed!=int(masks[:,1:length].sum()): raise RuntimeError('target accounting mismatch')
 return total,observed
def main():
 p=argparse.ArgumentParser(); p.add_argument('--source-checkpoint',type=Path,required=True); p.add_argument('--source-args-json',type=Path,required=True); p.add_argument('--source-weight-mode',choices=('train','saved'),default='train'); p.add_argument('--authority-root',type=Path,required=True); p.add_argument('--authority-sha256',required=True); p.add_argument('--pack-root',type=Path,required=True); p.add_argument('--pack-sha256',required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--steps',type=int,default=8); p.add_argument('--lr',type=float,default=1e-5); p.add_argument('--weight-decay',type=float,default=.01); p.add_argument('--sampler-key',type=int,default=71001); p.add_argument('--save-optimizer',action='store_true'); p.add_argument('--log-jsonl',type=Path,required=True); a=p.parse_args()
 dist.init_process_group('nccl'); rank=dist.get_rank(); world=dist.get_world_size(); torch.cuda.set_device(int(os.environ['LOCAL_RANK'])); device=torch.device('cuda')
 if world!=8: raise RuntimeError('dense agent qualification requires exactly eight ranks')
 if sha256(a.authority_root/'manifest.json')!=a.authority_sha256 or sha256(a.pack_root/'manifest.json')!=a.pack_sha256: raise RuntimeError('dataset authority mismatch')
 loaded=load_e97_checkpoint(a.source_checkpoint,args_json=a.source_args_json,device=device,dtype=torch.bfloat16,weight_mode=a.source_weight_mode,use_triton=True,mmap=False); model=loaded.model.train(); model.gradient_checkpointing=True; model.gradient_checkpoint_group_size=1
 optimizer=AdamWScheduleFree(model.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=a.weight_decay,warmup_steps=0); optimizer.train()
 identity=SFTSamplerIdentity(authority_manifest_sha256=a.authority_sha256,pack_manifest_sha256=a.pack_sha256,sampler_key=a.sampler_key,data_world_size=world,context_size=4096)
 data=MaskedSFTPackedDataset(a.authority_root,a.pack_root,identity=identity,rank=rank,verify_payload_hashes=True); a.output_root.mkdir(parents=True,exist_ok=True); a.log_jsonl.parent.mkdir(parents=True,exist_ok=True)
 emit(a.log_jsonl,'start',source_checkpoint=str(loaded.checkpoint_path),source_step=loaded.step,source_weight_mode=a.source_weight_mode,world_size=world,lr=a.lr,total_parameters=sum(x.numel() for x in model.parameters()))
 total_tokens=total_targets=0
 for update in range(1,a.steps+1):
  begin=time.monotonic(); optimizer.zero_grad(set_to_none=True); tokens,masks,lengths,target_counts,spans=data.get_batch_with_record_spans(1,device=device); local_targets=target_counts.sum().to(torch.int64); global_targets=local_targets.clone(); dist.all_reduce(global_targets,op=dist.ReduceOp.SUM)
  local_sum,_=objective(model,tokens,masks,int(lengths[0]),spans[0],global_targets.to(torch.float32)); sync_grads(model); grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
  counts=torch.tensor([int(lengths.sum()),int(local_targets),],device=device,dtype=torch.int64); dist.all_reduce(counts,op=dist.ReduceOp.SUM); loss_sum=local_sum.clone(); dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM); total_tokens+=int(counts[0]); total_targets+=int(counts[1]); emit(a.log_jsonl,'step',update=update,loss=float(loss_sum/global_targets),global_tokens=int(counts[0]),global_targets=int(counts[1]),total_tokens=total_tokens,total_targets=total_targets,grad_norm=float(grad_norm),step_seconds=time.monotonic()-begin,max_hbm_allocated=torch.cuda.max_memory_allocated())
 dist.barrier(); optimizer.eval()
 if rank==0:
  payload={'schema':'emender-e97-dense-agent-sft-v1','model_state_dict':model.state_dict(),'step':loaded.step+a.steps,'sft_updates':a.steps,'source_checkpoint':str(loaded.checkpoint_path),'source_step':loaded.step,'source_args_json':str(a.source_args_json),'source_weight_mode':a.source_weight_mode,'authority_manifest_sha256':a.authority_sha256,'pack_manifest_sha256':a.pack_sha256,'sampler_cursor':a.steps,'total_tokens':total_tokens,'assistant_target_tokens':total_targets,'weight_mode':'saved-eval'}
  if a.save_optimizer: payload['optimizer_state_dict']=optimizer.state_dict()
  out=a.output_root/f'checkpoint_agent_sft_u{a.steps:06d}.pt'; tmp=out.with_suffix('.pt.partial'); torch.save(payload,tmp); tmp.replace(out); emit(a.log_jsonl,'complete',checkpoint=str(out),checkpoint_bytes=out.stat().st_size,checkpoint_sha256=sha256(out))
 dist.barrier(); dist.destroy_process_group()
if __name__=='__main__': main()
