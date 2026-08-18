#!/usr/bin/env python3
"""Build a tiny immutable masked-SFT authority from paired-eval responses."""
from __future__ import annotations
import argparse, json, struct
from pathlib import Path
import tiktoken, torch
from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256


def entry(path): return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--panel',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--examples',type=int,default=8); a=p.parse_args()
    if a.examples<=0: raise SystemExit('examples must be positive')
    a.output_root.mkdir(parents=True,exist_ok=False)
    panel=torch.load(a.panel,map_location='cpu',weights_only=False); rows=panel['assistant_response_likelihood'][:a.examples]
    enc=tiktoken.get_encoding(panel['tokenizer']); paths={n:a.output_root/f for n,f in [('tokens','tokens.uint32.bin'),('mask','assistant_mask.uint8.bin'),('index','records.idx'),('metadata','records.jsonl')]}
    offset=total=targets=0
    with paths['tokens'].open('wb') as to, paths['mask'].open('wb') as mo, paths['index'].open('wb') as io, paths['metadata'].open('w') as meta:
      for row in rows:
        prefix=enc.encode_ordinary(row['prompt']); full=enc.encode_ordinary(row['prompt']+row['response'])
        if full[:len(prefix)]!=prefix: raise RuntimeError('response changes prompt tokenization')
        mask=[0]*len(prefix)+[1]*(len(full)-len(prefix)); to.write(struct.pack(f'<{len(full)}I',*full)); mo.write(bytes(mask)); io.write(RECORD_INDEX.pack(offset,len(full),sum(mask),0)); meta.write(json.dumps({'id':row['id'],'prompt':row['prompt'],'response':row['response']},sort_keys=True)+'\n'); offset+=len(full); total+=len(full); targets+=sum(mask)
    manifest={'schema':AUTHORITY_SCHEMA,'status':'complete','purpose':'isolated-conversation-overfit-diagnostic-only','examples':len(rows),'tokens':total,'assistant_target_tokens':targets,'source_panel_sha256':sha256(a.panel),'outputs':{n:entry(x) for n,x in paths.items()}}
    mp=a.output_root/'manifest.json'; mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    probe_panel=dict(panel); probe_panel['generation_prompts']=[{'id':row['id'],'template':'overfit-probe','prompt':row['prompt']} for row in rows]; probe_panel['overfit_references']={row['id']:row['response'] for row in rows}; probe_panel_path=a.output_root/'overfit_panel.pt'; torch.save(probe_panel,probe_panel_path)
    print(json.dumps({'manifest_sha256':sha256(mp),'overfit_panel_sha256':sha256(probe_panel_path),**manifest},sort_keys=True))
if __name__=='__main__': main()
