#!/usr/bin/env python3
"""Full byte-integrity and long-record validation for E97 instruction corpus."""
from __future__ import annotations
import argparse, codecs, hashlib, json, mmap, multiprocessing as mp
from pathlib import Path
import time
import tiktoken

_MM=None; _FILE=None; _ENC=None

def init_long(path):
 global _MM,_FILE,_ENC
 _FILE=open(path,'rb'); _MM=mmap.mmap(_FILE.fileno(),0,access=mmap.ACCESS_READ)
 _ENC=tiktoken.get_encoding('p50k_base')

def count_long(span):
 a,b=span; text=_MM[a:b].decode('utf-8','strict'); return len(_ENC.encode(text,disallowed_special=()))

def scan(path, collect_offsets=False):
 h=hashlib.sha256(); decoder=codecs.getincrementaldecoder('utf-8')('strict'); rs=0; size=0
 offsets=[0] if collect_offsets else None
 with path.open('rb') as f:
  while True:
   block=f.read(64*1024*1024)
   if not block:break
   h.update(block); decoder.decode(block,final=False); size+=len(block); rs+=block.count(b'\x1e')
  decoder.decode(b'',final=True)
 if collect_offsets:
  with path.open('rb') as f:
   mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ); start=0
   while True:
    pos=mm.find(b'\x1e',start)
    if pos<0:break
    offsets.append(pos+1); start=pos+1
   spans=[(offsets[i],offsets[i+1]-1) for i in range(len(offsets)-1)]
   spans.append((offsets[-1],len(mm))); mm.close()
  return size,rs,h.hexdigest(),spans
 return size,rs,h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
 manifest=json.loads((a.root/'e97_instruction_50b_v1.manifest.json').read_text())
 main_path=a.root/'e97_instruction_50b_v1.txt'; long_path=a.root/'e97_instruction_50b_v1_long32k.txt'
 ms,mr,mh=scan(main_path); ls,lr,lh,spans=scan(long_path,True)
 assert (ms,mr,mh)==(manifest['main']['file_bytes'],manifest['main']['rs_count'],manifest['main']['sha256'])
 assert (ls,lr,lh)==(manifest['long32k']['file_bytes'],manifest['long32k']['rs_count'],manifest['long32k']['sha256'])
 assert len(spans)==manifest['long32k']['records'] and all(b>a for a,b in spans)
 ctx=mp.get_context('fork')
 with ctx.Pool(a.workers,initializer=init_long,initargs=(str(long_path),)) as pool:
  lengths=list(pool.imap(count_long,spans,chunksize=16))
 assert min(lengths)>=32768
 # Exercise unchanged online-tokenization loader at all qualified contexts.
 from ndm.data.tokenized_dataset import TokenizedStreamDataset
 sampled={}
 for context in (2048,32768,131072):
  ds=TokenizedStreamDataset(str(main_path),context+1,seed=42,tokenizer_name='p50k_base')
  try:
   values,_reset,length=ds[0]; assert tuple(values.shape)==(context+1,); assert length==context+1; sampled[str(context)]=int(values.numel())
  finally: ds.close()
 receipt={'schema':'emender-e97-instruction-validation-v1','created_unix':time.time(),
  'main':{'bytes':ms,'rs_count':mr,'records':mr+1,'sha256':mh},
  'long32k':{'bytes':ls,'rs_count':lr,'records':lr+1,'sha256':lh,
    'min_tokens':min(lengths),'max_tokens':max(lengths)},
  'online_tokenization_sample_lengths':sampled,'status':'pass'}
 out=a.root/'e97_instruction_50b_v1.validation.json';out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
