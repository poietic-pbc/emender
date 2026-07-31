# E97 1.3B Pretrained 8-Node K-Sweep Synthesis - 2026-06-25

WG task: `synthesize-e97-1-3b`

## Decision

Do not promote any 8-node pretrained cadence directly to 16 nodes yet.

The K40 arm is the cleanest smoke-gated row: it completed reliably, finalized a
valid consensus checkpoint, and was the only arm whose fixed source-vs-candidate
eval delta stayed inside the original non-regression thresholds. However, that
fixed eval scored only 16,384 tokens. It is useful as a deterministic smoke
check, but it is too small to dominate the scale decision by itself.

The larger rank-0 training-loss windows make the bracket less one-sided:

- K160 is train-strong at its own longer endpoint, with the best endpoint
  last-500 and last-1000 local-step averages.
- K40 is best when all arms are compared over the same horizon through the K40
  endpoint, which is the fairest short-run cadence comparison.
- K320 is weak on both the tiny fixed eval and the larger training windows.

Recommendation: run a larger deterministic source-vs-candidate eval over K40
and K160, with K320 optional as a negative-control row if cheap, before choosing
any 16-node scale path. The blocker for immediate scaleout is that the only
quality-preserving signal favoring K40 is the tiny fixed-eval smoke slice, while
the larger train-loss aggregates give K160 credible counter-evidence at its own
endpoint.

No 16-node Slurm job was submitted from this synthesis task.

`run-64-node-e97` was checked after synthesis and remains `open (PAUSED)`.

## Shared Source And Baseline

Validation source: `docs/FRONTIER_E97_1P3B_PRETRAINED_VALIDATION_20260625.md`.

The validation task authorized only the bounded 8-node pretrained
GPU-island/no-DDP K40/K160/K320 bracket and explicitly excluded 16/32/64-node
continuation, GDN2/CMAES, and schedule-free outer jobs at that stage.

All rows below used the same staged source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt
```

All fixed eval comparisons used the same smoke scoring tensor:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

The source fixed eval baseline was:

| Row | Step | CE | BPB |
| --- | ---: | ---: | ---: |
| Source pretrained checkpoint | `260500` | `26.93483090` | `11.20646537` |

The absolute CE/BPB values on this 16,384-token smoke tensor are poor; the
useful signal is only the row-matched candidate-minus-source delta under the
same invocation. Because the tensor is tiny, the deltas are treated as smoke
evidence rather than final quality ranking evidence.

## Bracket Summary

| Row | Job ids | Nodes / topology | K / outer | Source -> final step | Train loss trend | Throughput | Merges | Node-hours | Fixed eval CE / BPB delta | Finalization behavior | Verdict |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | --- | --- |
| Source baseline | eval `4900725` | 1-node eval only | n/a | `260500` | Baseline checkpoint loss metadata `2.7480917453765867`; no continuation | n/a | n/a | eval elapsed `00:00:49` | `+0.00000000` / `+0.00000000` | Strict load OK; baseline CSV written | Baseline |
| 8-node K40 | train `4900838`; eval `4901316` | 8 Frontier nodes, 64 singleton GPU islands, no DDP | `DILOCO_K=40`, avg outer, export basis `x` | `260500 -> 263082` | Rank-0 first-20 mean `2.7153`, last-20 mean `2.6672`, `FINAL_LOSS_LAST100=2.6645`; endpoint last-500 `2.664524`, last-1000 `2.685093`, last-2000 `2.681430`; same-horizon last-1000 through new step 2580 `2.685093` | mean `139802` global tok/s; median after first 10 rows `164697` global tok/s | `66` | train requested/actual `8.0` / `6.704444`; eval requested/actual `1.0` / `0.023333` | `+0.00803614` / `+0.00334350` | Clean exit; periodic K-aligned saves through `263080`; final consensus merge at `263082`; final checkpoint and `latest.pt` valid | Provisionally clean; needs larger eval before scaleout |
| 8-node K160 | train `4900869`; eval `4901464` | 8 Frontier nodes, 64 singleton GPU islands, no DDP | `DILOCO_K=160`, avg outer, export basis `x` | `260500 -> 263840` | First-100 mean `2.71014`, last-100 mean `2.636824`, `FINAL_LOSS_LAST100=2.6368`; endpoint last-500 `2.636824`, last-1000 `2.676680`, last-2000 `2.684321`; same-horizon last-1000 through new step 2580 `2.710359` | filtered mean `167306`, filtered median `168217` global tok/s | `21` | train requested/actual `8.0` / `6.700`; eval actual `0.023056` from `00:01:23` elapsed | `+0.23717952` / `+0.09868055` | Clean exit; final step exactly on K boundary, final merge skipped because already consensus; final checkpoint and `latest.pt` valid | Train-strong at own endpoint, but smoke eval regressed; needs larger eval |
| 8-node K320 | train `4901367`; eval `4901744` | 8 Frontier nodes, 64 singleton GPU islands, no DDP | `DILOCO_K=320`, avg outer, export basis `x` | `260500 -> 263947` | First-20 mean `2.7035`, last-20 mean `2.5451`, `FINAL_LOSS_LAST100=2.6770`; endpoint last-500 `2.676967`, last-1000 `2.713588`, last-2000 `2.722002`; same-horizon last-1000 through new step 2580 `2.744789` | mean `161693`, median `165734`, last-20 mean `166255` global tok/s | `11` | train requested/actual `8.000000` / `6.704444`; eval actual `0.023333` | `+0.30940366` / `+0.12873002` | Clean exit; final consensus merge/checkpoint at `263947`; `latest.pt` valid; 64/64 final-ready markers present | Not a scale candidate from this bracket |

## Systems Interpretation

The systems result alone does not explain the quality split. All three 8-node
arms launched and finalized successfully, used the intended singleton-island
no-DDP path, wrote final checkpoints, and had no observed OOM, Python traceback,
non-finite loss, RCCL/NCCL watchdog timeout, or collective mismatch.

The cadence difference mostly changed merge frequency and synchronization cost:

- K40 merged `66` times, with total sync time `273.306` seconds and average
  sync `4141.0` ms.
- K160 merged `21` times, with total sync time `85.970` seconds and average
  sync `4093.8` ms.
- K320 merged `11` times, with total sync time `49.402` seconds and average
  sync `4491.1` ms.

Throughput for K160/K320 was higher than K40 because there were fewer save/merge
dips. K160 therefore has an operational advantage if a larger eval shows that
its apparent smoke-eval regression is not representative.

## Quality Interpretation

The original fixed eval gate was valuable for catching obvious deterministic
regressions, but it was only a tiny 16,384-token smoke slice. It should not be
used as the sole ranking signal.

The loss aggregates show two different stories:

- Endpoint comparison favors K160:
  - last 500 local steps: K40 `2.664524`, K160 `2.636824`, K320 `2.676967`
  - last 1000 local steps: K40 `2.685093`, K160 `2.676680`, K320 `2.713588`
  - last 2000 local steps: K40 `2.681430`, K160 `2.684321`, K320 `2.722002`
- Same-horizon comparison through K40's endpoint favors K40:
  - last 1000 local steps through new step 2580: K40 `2.685093`,
    K160 `2.710359`, K320 `2.744789`

This means K40 is the conservative same-horizon candidate, while K160 is a real
endpoint contender rather than a simple failed row. K320 remains the weakest
recipe because it loses on smoke eval and on the broader training-loss windows.

## Next Step

Create no immediate 16-node scaleout job from this synthesis. Instead, use the
8-node artifacts to run a larger deterministic fixed source-vs-candidate eval
before picking a scale recipe.

Recommended bounded eval follow-up:

- Score the staged source checkpoint, K40 final checkpoint, and K160 final
  checkpoint on the same larger deterministic token tensor.
- Include K320 only if the extra eval cost is negligible; it is useful as a
  negative-control row but not necessary for selecting between K40 and K160.
- Use identical evaluator code, model loading mode, tokenizer, batch size, and
  `--y-mode saved` semantics across rows.
- Report CE/BPB, candidate-minus-source deltas, token count, walltime,
  node-hours, checkpoint paths, and any strict-load/finalization anomalies.
- Select a 16-node recipe only if the larger eval and the train-loss windows
  agree that one row is quality-preserving enough to scale.

The previously opened downstream scaleout task `run-e97-1-3b-4` is no longer
the recommended next step under this corrected interpretation.

## Scope Confirmation

- Read and compared `docs/FRONTIER_E97_1P3B_PRETRAINED_VALIDATION_20260625.md`.
- Read and compared all completed 8-node K reports:
  - `docs/FRONTIER_E97_1P3B_PRETRAINED_8NODE_K40_20260625.md`
  - `docs/FRONTIER_E97_1P3B_PRETRAINED_K160_8NODE_20260625.md`
  - `docs/FRONTIER_E97_1P3B_PRETRAINED_8N_K320_20260625.md`
- Incorporated the correction that fixed eval is a tiny 16,384-token smoke slice
  and should not dominate interpretation.
- Labeled K160 train-strong at its own endpoint and K40 favored only on the
  same-horizon K40-endpoint comparison.
- Did not run `sbatch` and submitted no Slurm job from this synthesis task.
- Confirmed `run-64-node-e97` remains `open (PAUSED)`.
