#!/usr/bin/env python3
"""Evaluate a pinned Hugging Face causal LM on the E97 MMLU/HellaSwag panel."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True); p.add_argument('--revision',required=True)
    p.add_argument('--panel',type=Path,required=True); p.add_argument('--panel-sha256',required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if sha256(a.panel)!=a.panel_sha256: raise RuntimeError('panel SHA-256 mismatch')
    panel=torch.load(a.panel,map_location='cpu',weights_only=False)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    letters='ABCD'; ids=[]
    for x in letters:
        v=tok.encode(' '+x,add_special_tokens=False)
        if len(v)!=1: raise RuntimeError('answer letter is not one token')
        ids.append(v[0])
    mmlu=[]
    with torch.inference_mode():
      for e in panel['mmlu']:
        choices='\n'.join(f'{letters[i]}. {c}' for i,c in enumerate(e['choices']))
        prompt=f"Question: {e['question']}\n{choices}\nAnswer:"
        x=tok(prompt,return_tensors='pt').input_ids.cuda(); logits=model(x).logits[0,-1].float()
        scores=logits[ids].cpu().tolist(); pred=max(range(4),key=lambda i:scores[i])
        mmlu.append({'id':e['id'],'answer':e['answer'],'prediction':pred,'correct':pred==e['answer'],'scores':scores})
    hs=[]
    with torch.inference_mode():
      for e in panel['hellaswag']:
        raw=[]; norm=[]; lengths=[]
        prefix=tok.encode(e['context'],add_special_tokens=False)
        for c in e['choices']:
          c=c if c.startswith(' ') else ' '+c
          full=tok.encode(e['context']+c,add_special_tokens=False)
          if full[:len(prefix)]!=prefix: raise RuntimeError('continuation prefix changed')
          x=torch.tensor(full,device='cuda').unsqueeze(0); logits=model(x).logits
          target=x[0,len(prefix):]; selected=logits[0,len(prefix)-1:len(full)-1].float()
          losses=F.cross_entropy(selected,target,reduction='none'); raw.append(float(-losses.sum())); norm.append(float(-losses.mean())); lengths.append(len(target))
        rp=max(range(4),key=lambda i:raw[i]); np=max(range(4),key=lambda i:norm[i])
        hs.append({'id':e['id'],'answer':e['answer'],'raw_prediction':rp,'normalized_prediction':np,'raw_correct':rp==e['answer'],'normalized_correct':np==e['answer'],'raw_scores':raw,'normalized_scores':norm,'choice_tokens':lengths})
    preds=[sum(r['prediction']==i for r in mmlu) for i in range(4)]
    out={'schema':'emender-hf-causal-baseline-v1','model':str(a.model),'revision':a.revision,'panel_sha256':a.panel_sha256,'mmlu':mmlu,'hellaswag':hs,'summary':{'mmlu_accuracy':sum(r['correct'] for r in mmlu)/len(mmlu),'hellaswag_accuracy':sum(r['raw_correct'] for r in hs)/len(hs),'hellaswag_normalized_accuracy':sum(r['normalized_correct'] for r in hs)/len(hs),'mmlu_prediction_counts':preds}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out['summary'],sort_keys=True))
if __name__=='__main__': main()
