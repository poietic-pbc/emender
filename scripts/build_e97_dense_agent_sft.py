#!/usr/bin/env python3
"""Build deterministic, mechanically verified tool-use traces for dense E97."""
from __future__ import annotations
import argparse, hashlib, json, random, struct
from pathlib import Path
import tiktoken
from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
RS='\x1e'; ENC='p50k_base'
SYSTEM=('You are a precise tool-using agent. Respond with either "Action:" and one JSON '
        '"Arguments:" object, or "Final:". Never invent tool results.')

def split(identity): return 1 if int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'little')%100==0 else 0
def entry(p): return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha256(p)}
def trace(kind,i,rng):
 if kind=='calculator':
  a=rng.randint(11,999); b=rng.randint(11,999); op=rng.choice(['+','-','*']); value=eval(f'{a}{op}{b}'); expr=f'{a} {op} {b}'
  return f'Calculate {expr}.', [('assistant',f'Action: calculator\nArguments: {{"expression":"{expr}"}}'),('tool',json.dumps({'result':str(value)},separators=(',',':'))),('assistant',f'Final: {expr} = {value}.')]
 if kind=='lookup':
  project=f'Project-{i:06d}'; owner=rng.choice(['Amina','Boris','Chen','Devika','Elena','Farid']); budget=rng.randint(20,900)*1000; field=rng.choice(['owner','budget'])
  answer=owner if field=='owner' else f'${budget:,}'
  content=f'{project} has owner {owner} and approved budget ${budget:,}.'
  return f'What is the {field} of {project}? Use the records.', [('assistant',f'Action: search\nArguments: {{"query":"{project}"}}'),('tool',json.dumps({'matches':[f'records/{project}.txt']},separators=(',',':'))),('assistant',f'Action: read\nArguments: {{"path":"records/{project}.txt"}}'),('tool',json.dumps({'content':content},separators=(',',':'))),('assistant',f'Final: The {field} of {project} is {answer}.')]
 if kind=='count':
  ext=rng.choice(['.md','.py','.json','.txt']); n=rng.randint(2,18); files=[f'data/item-{i}-{j}{ext}' for j in range(n)]
  return f'How many {ext} files are in data/?', [('assistant',f'Action: list\nArguments: {{"path":"data","suffix":"{ext}"}}'),('tool',json.dumps({'files':files},separators=(',',':'))),('assistant',f'Final: There are {n} {ext} files in data/.')]
 raise ValueError(kind)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--records',type=int,default=30000); p.add_argument('--seed',type=int,default=9701); a=p.parse_args(); a.output_root.mkdir(parents=True,exist_ok=False); enc=tiktoken.get_encoding(ENC)
 paths={n:a.output_root/f for n,f in [('tokens','tokens.uint32.bin'),('mask','assistant_mask.uint8.bin'),('index','records.idx'),('metadata','records.jsonl')]}; counts={'records':0,'tokens':0,'assistant_target_tokens':0,'train_records':0,'validation_records':0}; offset=0
 with paths['tokens'].open('wb') as to, paths['mask'].open('wb') as mo, paths['index'].open('wb') as io, paths['metadata'].open('w') as meta:
  for i in range(a.records):
   kind=('calculator','lookup','count')[i%3]; identity=f'agent-{kind}-{i:08d}'; user,turns=trace(kind,i,random.Random(a.seed+i)); messages=[('system',SYSTEM),('user',user),*turns]; pieces=[]
   for j,(role,text) in enumerate(messages):
    if j: pieces.append(('\n\n',False))
    pieces.append(({'system':'System','user':'User','assistant':'Assistant','tool':'Tool'}[role]+':\n',False)); pieces.append((text,role=='assistant'))
    if role=='assistant': pieces.append((RS,True))
   complete=''.join(x for x,_ in pieces); target_ranges=[]; cursor=0
   for text,target in pieces:
    stop=cursor+len(text.encode());
    if target: target_ranges.append((cursor,stop))
    cursor=stop
   toks=enc.encode_ordinary(complete); masks=[]; decoded=bytearray()
   for tok in toks:
    left=len(decoded); bs=enc.decode_single_token_bytes(tok); decoded.extend(bs); right=len(decoded); overlaps=[(x,y) for x,y in target_ranges if left<y and right>x]
    if not overlaps: masks.append(0)
    elif any(left>=x and right<=y for x,y in overlaps): masks.append(1)
    else: raise RuntimeError('token crosses target boundary')
   s=split(identity); to.write(struct.pack(f'<{len(toks)}I',*toks)); mo.write(bytes(masks)); io.write(RECORD_INDEX.pack(offset,len(toks),sum(masks),s)); meta.write(json.dumps({'id':identity,'source':f'emender-agent-{kind}-v1','split':s,'tokens':len(toks),'targets':sum(masks)},sort_keys=True)+'\n'); offset+=len(toks); counts['records']+=1; counts['tokens']+=len(toks); counts['assistant_target_tokens']+=sum(masks); counts['validation_records' if s else 'train_records']+=1
 manifest={'schema':AUTHORITY_SCHEMA,'status':'complete','purpose':'dense-e97-bounded-tool-agent-v1','serialization':'System/User/Assistant/Tool with RS after every assistant turn','seed':a.seed,'counts':counts,'outputs':{n:entry(x) for n,x in paths.items()}}; mp=a.output_root/'manifest.json'; mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); print(json.dumps({'manifest_sha256':sha256(mp),**manifest},sort_keys=True))
if __name__=='__main__': main()
