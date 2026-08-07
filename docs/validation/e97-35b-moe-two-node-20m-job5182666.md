# E97 35B MoE two-node 20-minute DiLoCo qualification — job 5182666

Source `da46d082`; final 513B E97 seed; two eight-GCD node islands.
Scheduler evidence: `Partition=batch`, `QOS=debug`. Terminal state
`COMPLETED`, exit `0:0`, elapsed `00:28:21`.

- measured training: 1,200.738 seconds
- steps: 82
- accepted tokens: 2,686,976
- mean loss: 2.661326
- median throughput: 2,362.63 tokens/s
- maximum allocated HBM: 48,729,292,800 bytes/GCD
- K40 DiLoCo merges: 17.206 s and 13.517 s

Expert-token assignment/return remained inside each separately proven
node-local eight-rank group. Only corresponding model and ScheduleFree `x/z`
shards crossed nodes during the two K40 DiLoCo merges. All steps and both
post-merge continuations remained finite.

JSONL: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/2-node-20m-5182666/training.jsonl`

This is a pre-production qualification observation, not an ADR-003 production
rung; ADR-003's production predecessor ladder begins at eight nodes.
