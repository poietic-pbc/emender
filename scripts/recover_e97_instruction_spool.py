#!/usr/bin/env python3
"""Finish a complete E97 instruction shuffle spool after walltime interruption."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import time
import numpy as np

from build_e97_instruction_corpus import INDEX_DTYPE, emit, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--inventory-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--readme',type=Path)
    args=p.parse_args()
    spec=json.loads(args.spec.read_text())
    spool=args.output_root/'shuffle-spool'
    main_paths=sorted(spool.glob('main-*.bin'))
    long_paths=sorted(spool.glob('long-*.bin'))
    if not main_paths or len(main_paths)!=len(long_paths):
        raise SystemExit('complete paired shuffle spool is unavailable')
    buckets=len(main_paths)
    rng=np.random.default_rng(int(spec['seed']))
    selections=[]; long_candidates=[]
    for pool_id,source in enumerate(spec['sources']):
        name=source['name']; index=np.fromfile(args.inventory_root/f'{name}.index',dtype=INDEX_DTYPE)
        candidates=np.arange(len(index),dtype=np.int64)
        target=int(source['target_tokens']); actual=records=payload_bytes=epochs=0
        while actual<target:
            epochs+=1; permutation=rng.permutation(len(candidates))
            contribution=index['tokens'][candidates[permutation]].astype(np.uint64)+1
            take=min(len(permutation),int(np.searchsorted(np.cumsum(contribution,dtype=np.uint64),target-actual))+1)
            chosen=candidates[permutation[:take]]
            actual+=int(contribution[:take].sum(dtype=np.uint64)); records+=take
            payload_bytes+=int(index['bytes'][chosen].sum(dtype=np.uint64))
            # Reproduce one bucket draw per selected occurrence.
            rng.integers(buckets,size=take)
        selections.append({'source':name,'target_tokens':target,
            'actual_contribution_tokens':actual,'payload_tokens':actual-records,
            'rs_tokens':records,'overshoot_tokens':actual-target,
            'selected_records':records,'selected_payload_bytes':payload_bytes,
            'unique_candidates':len(candidates),'epochs_started':epochs})
        eligible=np.flatnonzero(index['tokens']>=int(spec['long_context_min_tokens']))
        if len(eligible): long_candidates.append((name,index,eligible))
    union=[]
    for pool_id,(_name,index,eligible) in enumerate(long_candidates):
        union.extend((pool_id,int(i),int(index[int(i)]['tokens'])) for i in eligible)
    target=int(spec['long_context']['target_tokens']); actual=0; selected=[]
    while actual<target:
        item=union[int(rng.integers(len(union)))]; selected.append(item); actual+=item[2]+1
    payload_bytes=0
    for pool_id,row_id,_tokens in selected:
        payload_bytes+=int(long_candidates[pool_id][1][row_id]['bytes'])
    # Reproduce main and long bucket draws before emit uses the RNG.
    rng.integers(buckets,size=len(selected)); rng.integers(buckets,size=len(selected))
    long_receipt={'source':'long32k','target_tokens':target,
        'actual_contribution_tokens':actual,'payload_tokens':actual-len(selected),
        'rs_tokens':len(selected),'overshoot_tokens':actual-target,
        'selected_records':len(selected),'selected_payload_bytes':payload_bytes,
        'unique_candidates':len(union)}
    selections.append(long_receipt)
    main_output=args.output_root/'e97_instruction_50b_v1.txt'
    long_output=args.output_root/'e97_instruction_50b_v1_long32k.txt'
    main_receipt=emit(main_paths,main_output,rng)
    long_receipt_file=emit(long_paths,long_output,rng)
    main_tokens=sum(int(x['actual_contribution_tokens']) for x in selections)-1
    manifest={'schema':'emender-e97-instruction-corpus-v1','created_unix':time.time(),
      'recovered_from_complete_spool':True,'seed':spec['seed'],'delimiter_hex':'1e',
      'target_tokens':int(spec['target_tokens']),'main_accounted_tokens':main_tokens,
      'main_overshoot_tokens':main_tokens-int(spec['target_tokens']),
      'long32k_accounted_tokens':actual-1,'tokenizer':spec['tokenizer'],
      'tokenizer_sha256':spec['tokenizer_sha256'],
      'token_accounting':'p50k payload tokens plus one RS token per selected occurrence; final stream has one fewer RS',
      'selections':selections,'main':main_receipt,'long32k':long_receipt_file,
      'spec_sha256':hashlib.sha256(args.spec.read_bytes()).hexdigest()}
    write_json(args.output_root/'e97_instruction_50b_v1.manifest.json',manifest)
    (args.output_root/'e97_instruction_50b_v1.sources.json').write_bytes(args.spec.read_bytes())
    if args.readme:(args.output_root/'README.md').write_bytes(args.readme.read_bytes())
    (args.output_root/'e97_instruction_50b_v1.sha256').write_text(
      f"{main_receipt['sha256']}  {main_output.name}\n{long_receipt_file['sha256']}  {long_output.name}\n")
    for path in main_paths+long_paths:path.unlink()
    spool.rmdir()
    print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__':main()
