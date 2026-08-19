#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--shard-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); shards=[json.loads(x.read_text()) for x in sorted(a.shard_dir.glob('shard-*.json'))]; rows=[r for s in shards for r in s['rows']]; turns=[r for r in rows if not r.get('task_summary')]; tasks=[r for r in rows if r.get('task_summary')]; actions=[r for r in turns if r['turn_type']=='action']; finals=[r for r in turns if r['turn_type']=='final']; frac=lambda xs,k:sum(bool(x[k]) for x in xs)/len(xs) if xs else 0
 out={'schema':'emender-e97-dense-agent-eval-v1','checkpoint':shards[0]['checkpoint'],'shards':len(shards),'tasks':len(tasks),'turns':len(turns),'task_exact_accuracy':frac(tasks,'exact'),'turn_exact_accuracy':frac(turns,'exact'),'action_exact_accuracy':frac(actions,'exact'),'action_syntax_validity':frac(actions,'action_valid'),'final_exact_accuracy':frac(finals,'exact'),'stop_accuracy':frac(turns,'stopped'),'by_kind':{}}
 for kind in sorted({r['kind'] for r in tasks}):
  kt=[r for r in tasks if r['kind']==kind]; kr=[r for r in turns if r['kind']==kind]; out['by_kind'][kind]={'tasks':len(kt),'task_exact_accuracy':frac(kt,'exact'),'turn_exact_accuracy':frac(kr,'exact')}
 a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True))
if __name__=='__main__': main()
