#!/usr/bin/env python3
"""Score one deterministic shard of full HellaSwag for E97 or a pinned HF LM."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=('e97','hf'),required=True)
    p.add_argument('--model',type=Path,required=True); p.add_argument('--args-json',type=Path)
    p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True); p.add_argument('--world',type=int,required=True)
    p.add_argument('--label',required=True); p.add_argument('--weight-mode',choices=('train','saved'),default='train')
    return p.parse_args()


def main():
    a=args(); torch.cuda.set_device(0)
    if not 0<=a.rank<a.world: raise RuntimeError('invalid shard rank')
    rows=pq.read_table(a.data).to_pylist()
    if a.backend=='e97':
        import tiktoken
        from ndm.e97 import load_e97_checkpoint
        enc=tiktoken.get_encoding('p50k_base'); encode=enc.encode_ordinary
        loaded=load_e97_checkpoint(a.model,args_json=a.args_json,device='cuda',dtype=torch.bfloat16,weight_mode=a.weight_mode,use_triton=True,mmap=True)
        model=loaded.model.eval()
        def forward(x,mask): return model(x)
        identity={'step':loaded.step,'weight_mode':loaded.weight_mode,'schedulefree_train_weight_swap':loaded.schedulefree_train_weight_swap}
    else:
        from transformers import AutoModelForCausalLM,AutoTokenizer
        tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); encode=lambda s:tok.encode(s,add_special_tokens=False)
        model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
        def forward(x,mask): return model(x,attention_mask=mask).logits
        identity={}
    records=[]
    with torch.inference_mode():
      for index in range(a.rank,len(rows),a.world):
        e=rows[index]; context=e['ctx']; prefix=encode(context); sequences=[]; lengths=[]
        for ending in e['endings']:
          continuation=ending if ending.startswith(' ') else ' '+ending
          full=encode(context+continuation)
          if full[:len(prefix)]!=prefix or len(full)<=len(prefix): raise RuntimeError('continuation tokenization changed prefix')
          sequences.append(full); lengths.append(len(full))
        width=max(lengths); x=torch.zeros((4,width),dtype=torch.long,device='cuda'); mask=torch.zeros_like(x)
        for i,seq in enumerate(sequences): x[i,:len(seq)]=torch.tensor(seq,device='cuda'); mask[i,:len(seq)]=1
        logits=forward(x,mask).float(); raw=[]; norm=[]; counts=[]
        for i,length in enumerate(lengths):
          target=x[i,len(prefix):length]; selected=logits[i,len(prefix)-1:length-1]
          losses=F.cross_entropy(selected,target,reduction='none'); raw.append(float(-losses.sum())); norm.append(float(-losses.mean())); counts.append(len(target))
        answer=int(e['label']); rp=max(range(4),key=lambda i:raw[i]); np=max(range(4),key=lambda i:norm[i])
        records.append({'index':index,'answer':answer,'raw_prediction':rp,'normalized_prediction':np,'raw_correct':rp==answer,'normalized_correct':np==answer,'raw_scores':raw,'normalized_scores':norm,'choice_tokens':counts})
        if len(records)%100==0: print(json.dumps({'event':'progress','label':a.label,'rank':a.rank,'complete':len(records)}),flush=True)
    out={'schema':'emender-full-hellaswag-shard-v1','label':a.label,'backend':a.backend,'rank':a.rank,'world':a.world,'examples_total':len(rows),'model':str(a.model),'identity':identity,'records':records}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,sort_keys=True)+'\n')
    print(json.dumps({'event':'complete','label':a.label,'rank':a.rank,'records':len(records)}),flush=True)
if __name__=='__main__': main()
