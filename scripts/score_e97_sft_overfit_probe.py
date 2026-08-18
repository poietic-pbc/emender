#!/usr/bin/env python3
"""Score exact-prompt overfit generations against immutable references."""
from __future__ import annotations
import argparse, json
from difflib import SequenceMatcher
from pathlib import Path
import torch

def norm(s: str) -> str: return ' '.join(s.replace('\x1e','').split())
def main():
 p=argparse.ArgumentParser(); p.add_argument('--panel',type=Path,required=True); p.add_argument('--evaluation',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 refs=torch.load(a.panel,map_location='cpu',weights_only=False)['overfit_references']; result=json.loads(a.evaluation.read_text()); rows=[]
 for g in result['generations']:
  if g['mode']!='greedy': continue
  ref=norm(refs[g['id']]); got=norm(g['response']); rows.append({'id':g['id'],'exact':got==ref,'reference_prefix':bool(got) and ref.startswith(got),'generation_prefix':bool(ref) and got.startswith(ref),'similarity':SequenceMatcher(None,ref,got).ratio(),'stopped':g['stopped'],'generated_tokens':g['generated_tokens']})
 out={'schema':'emender-e97-sft-overfit-score-v1','evaluation':str(a.evaluation.resolve()),'examples':len(rows),'exact':sum(x['exact'] for x in rows),'mean_similarity':sum(x['similarity'] for x in rows)/len(rows),'stopped':sum(x['stopped'] for x in rows),'rows':rows}; a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True))
if __name__=='__main__': main()
