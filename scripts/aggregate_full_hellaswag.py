#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path

def interval(values,seed=970035):
 r=random.Random(seed); n=len(values); means=[]
 for _ in range(10000): means.append(sum(values[r.randrange(n)] for _ in range(n))/n)
 means.sort(); return {'mean':sum(values)/n,'p025':means[250],'p975':means[9750]}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--shards',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 payloads=[json.loads(x.read_text()) for x in sorted(a.shards.glob('rank-*.json'))]
 if len(payloads)!=8 or {x['rank'] for x in payloads}!=set(range(8)): raise RuntimeError('requires eight complete shards')
 records=sorted((r for x in payloads for r in x['records']),key=lambda r:r['index']); total=payloads[0]['examples_total']
 if len(records)!=total or [r['index'] for r in records]!=list(range(total)): raise RuntimeError('full coverage mismatch')
 out={'schema':'emender-full-hellaswag-v1','label':payloads[0]['label'],'backend':payloads[0]['backend'],'model':payloads[0]['model'],'identity':payloads[0]['identity'],'examples':total,'raw_accuracy':interval([int(r['raw_correct']) for r in records]),'normalized_accuracy':interval([int(r['normalized_correct']) for r in records]),'records':records}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({k:v for k,v in out.items() if k!='records'},sort_keys=True))
if __name__=='__main__': main()
