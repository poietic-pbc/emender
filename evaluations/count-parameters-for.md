# Count parameters for live E97 DiLoCo continuation

Date: 2026-06-23

## Answer

The live E97/Emender 8-GPU DiLoCo continuation has:

- Total unique model parameters: **1,286,589,072**
- Trainable parameters: **1,286,589,072**

All unique model parameters are trainable. The run is therefore a ~1.29B-parameter
model, not a 100M-parameter model.

## Method

I used two independent read-only checks:

1. The live run log at
   `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log` reports:
   `Model: Level E97, 1,286,589,072 parameters`.
2. I instantiated the exact configured model on CPU from the live recipe:
   `level=E97`, `dim=1792`, `depth=11`, `n_heads=216`, `n_state=32`,
   `expansion=1.0`, `use_gate=1`, `gate_activation=silu`,
   `mlp_ratio=2.2623`, `mlp_multiple=64`, `use_triton=1`,
   `tokenizer=p50k_base` (`vocab_size=50281`). The instantiated model reported
   `sum(p.numel() for p in model.parameters()) == 1,286,589,072` and the same
   count with `p.requires_grad`.

I also inspected the latest checkpoint metadata with `torch.load(...,
map_location="meta")`. The checkpoint `model_state_dict` tensor-entry sum was
`1,376,692,624`, which is larger by exactly `90,103,552` entries because the
tied embedding/language-head weight appears under both `embedding.weight` and
`lm_head.weight` in `state_dict`. That is a serialization alias/duplicate entry,
not an extra trainable parameter. The model's own `get_num_params()` and
`named_parameters()` count unique parameters and give **1,286,589,072**.

Useful breakdown from the CPU instantiation:

- Tied token embedding / LM head unique weight: `50,281 * 1,792 = 90,103,552`
- Pre-layer RMSNorms: `11 * 1,792 = 19,712`
- Final RMSNorm: `1,792`
- Recurrent/MLP layers: `1,196,464,016`
- Total unique parameters: `90,103,552 + 19,712 + 1,196,464,016 + 1,792 = 1,286,589,072`
- Buffers: `0`

## Why the directory says `100m`

The `100m` in
`/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742`
is the `--params` launcher/request label, not an exact measured or auto-derived
parameter count.

Grounding:

- The live `args.json` records `"params": "100m"`, while also recording the
  explicit shape overrides that define the actual model: `"dim": 1792`,
  `"depth": 11`, `"n_heads": 216`, `"n_state": 32`, and `"mlp_ratio": 2.2623`.
- `train.py` constructs the directory name as
  `f"level{args.level}_{args.params}_{timestamp}"`.
- Because `--dim` and `--depth` are explicitly supplied, `train.py` follows the
  explicit-shape `LadderLM(...)` construction path instead of the
  `create_ladder_model(target_params=args.params, ...)` approximate target-bucket
  path.

So `levelE97_100m` is a stale/approximate target bucket or launcher label kept in
the run name. It is not the exact size of this live model.

## Process-list numeric prefixes

Process-list numeric prefixes such as `907099` are PIDs, not model parameter
counts. For example, `ps -o pid,ppid,stat,cmd -p 906526,907099,934892,675799`
showed:

- `906526`: the `torchrun` leader PID.
- `907099`: one `train.py` worker process PID, child of `906526`.
- `934892`: the supervisor shell PID.
- `675799`: the watchdog shell PID.

The live job was not modified and no duplicate training run was launched.
