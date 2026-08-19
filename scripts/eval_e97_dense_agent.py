#!/usr/bin/env python3
"""Greedy held-out exact-turn evaluation for the bounded dense E97 agent."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from ndm.e97 import generate_e97,load_e97_checkpoint
from scripts.build_e97_dense_agent_sft import ENC,RS,SYSTEM,split,trace
ROLES={'system':'System','user':'User','assistant':'Assistant','tool':'Tool'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--args-json',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--rank',type=int,required=True); p.add_argument('--world',type=int,required=True); p.add_argument('--records',type=int,default=30000); p.add_argument('--seed',type=int,default=9701); a=p.parse_args(); torch.cuda.set_device(0)
 loaded=load_e97_checkpoint(a.checkpoint,args_json=a.args_json,device='cuda',dtype=torch.bfloat16,weight_mode='saved',use_triton=True,mmap=True); enc=__import__('tiktoken').get_encoding(ENC); rows=[]; validation_index=0
 for i in range(a.records):
  kind=('calculator','lookup','count')[i%3]; identity=f'agent-{kind}-{i:08d}'
  if split(identity)!=1: continue
  if validation_index%a.world!=a.rank: validation_index+=1; continue
  validation_index+=1; user,turns=trace(kind,i,__import__('random').Random(a.seed+i)); messages=[('system',SYSTEM),('user',user),*turns]; built=''; turn_index=0; task_exact=True
  for j,(role,text) in enumerate(messages):
   if j: built+='\n\n'
   built+=ROLES[role]+':\n'
   if role!='assistant': built+=text; continue
   expected=text+RS; generated=generate_e97(loaded,built,max_new_tokens=len(enc.encode_ordinary(expected))+8,temperature=0,top_k=0,max_context=4096,mode='full-context',stop_token_ids=(218,)); got=generated['new_text']; exact=got==expected; task_exact &= exact
   action_valid=False
   if text.startswith('Action:'):
    try:
     lines=got.replace(RS,'').splitlines(); action_valid=lines[0].startswith('Action: ') and lines[1].startswith('Arguments: ') and isinstance(json.loads(lines[1][11:]),dict)
    except Exception: pass
   rows.append({'id':identity,'kind':kind,'turn':turn_index,'turn_type':'action' if text.startswith('Action:') else 'final','expected':expected,'generated':got,'exact':exact,'stopped':bool(generated['new_token_ids'] and generated['new_token_ids'][-1]==218),'action_valid':action_valid}); built+=expected; turn_index+=1
  rows.append({'id':identity,'kind':kind,'task_summary':True,'exact':task_exact})
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'schema':'emender-e97-dense-agent-eval-shard-v1','rank':a.rank,'world':a.world,'checkpoint':str(a.checkpoint),'rows':rows},sort_keys=True)+'\n')
if __name__=='__main__': main()
