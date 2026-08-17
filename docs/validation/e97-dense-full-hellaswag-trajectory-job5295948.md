# Dense E97 full-HellaSwag trajectory audit

Job `5295948` completed on six nodes in parallel (`Partition=batch`, `QOS=debug`,
`Requeue=0`). Five immutable dense E97 checkpoints and pinned GPT-2 XL revision
`15ea56dee5df4983c59b2538573817e1667135e2` were each sharded over one node's
eight GCDs. Every system scored all 10,042 HellaSwag validation examples.

The initial job `5295827` failed before model evaluation because compute nodes
could not fetch the p50k tokenizer. Source `7d8cece3` bound and hash-verified the
canonical offline cache; the attended replacement then completed.

| System | Global exposure | Raw accuracy | Length-normalized accuracy |
|---|---:|---:|---:|
| dense E97 | 299.608B | 0.31936 | 0.36537 |
| dense E97 | 343.228B | 0.31936 | 0.36437 |
| dense E97 | 353.295B | 0.31936 | 0.36487 |
| dense E97 | 400.942B | 0.31826 | 0.36447 |
| dense E97 | 513.014B | 0.31846 | 0.36517 |
| GPT-2 XL control | n/a | 0.38379 | 0.48905 |

The p50k and GPT-2 tokenizers produce exactly identical token IDs for all 256
contexts and all 1,024 context-choice strings in the earlier panel; tokenizer
differences do not explain that panel. Full-set paired bootstrap deltas for
GPT-2 XL minus dense-513B are `+0.06533 [0.05935, 0.07150]` raw and
`+0.12388 [0.11571, 0.13205]` normalized.

Dense E97 made no measurable HellaSwag progress from 299.6B to 513.0B global
exposure: 513B minus 300B is `-0.00090 [-0.00299, 0.00110]` raw and
`-0.00020 [-0.00299, 0.00249]` normalized. This is not a small-panel artifact.
The additional 213B globally counted tokens did not translate into capability
on this benchmark.

The result does not alone distinguish architecture, data, and optimization.
The run used very large global token batches with K40 DiLoCo; global token
exposure is not equivalent to the number of sequential parameter updates or
per-worker trajectory tokens. Further SFT cannot resolve this foundation-level
plateau. No additional alignment training is authorized by this result.

Artifacts:

- `/lustre/orion/bif148/proj-shared/emender/evaluations/e97-dense-full-hellaswag-trajectory-v2/results`
- launcher/source: `7d8cece3`
