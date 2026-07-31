# E97 step 1,525,000 intake and launcher integration

Date: 2026-07-12

This report records the intake verification used to pin the immutable refreshed
seed in the canonical E97 256-node smoke/production renderer. No `sbatch`
command was executed during this integration task.

## Immutable S3 object and prefix

Object:
`s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_1525000/checkpoint_step_1525000_loss_2.4378.pt`

The concrete HTTP S3 HEAD returned:

```text
HTTP/1.1 200 OK
Last-Modified: Sun, 12 Jul 2026 15:16:53 GMT
ETag: "cf57a33307ab452cbad4d3a5daaa156f-116"
x-amz-server-side-encryption: AES256
Accept-Ranges: bytes
Content-Type: application/vnd.snesdev-page-table
Content-Length: 7719679924
Server: AmazonS3
```

The concrete S3 `list-type=2` request for the immutable step prefix returned
`KeyCount=14`, `IsTruncated=false`, and included the checkpoint with
`Size=7719679924`, its `.sha256`, `manifest.json`,
`latest_emender_E97_1.3B.json`, `loss_metrics.json`, and the retained metadata
checksum bundle. The dynamic latest manifest was fetched only in this intake
step. It is not fetched or resolved by the renderer, wrapper, or launch argv.

## Retained intake bundle

Retained directory:
`/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000`

Concrete `stat -c %s` stdout:

```text
7719679924
```

Concrete `sha256sum` stdout:

```text
1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9  /lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt
```

The upstream checksum file names the same digest. Concrete
`sha256sum -c metadata_files.sha256` stdout for the retained metadata bundle:

```text
args.json: OK
checkpoint_step_1525000_loss_2.4378.pt.sha256: OK
e97_diloco_loss_curve_smoothed_20260623.png: OK
latest_emender_E97_1.3B.json: OK
latest_symlink_target.txt: OK
launch_manifest.json: OK
loss_metrics.json: OK
manifest.json: OK
run_command_process_snapshot.txt: OK
run_log_tail_1000.txt: OK
run_manifest.json: OK
supervisor.log: OK
```

The retained manifest reports `step=1525000`, `loss=2.4378`,
`tokens=99942400000`, `tokens_billions=99.9424`,
`checkpoint_size_bytes=7719679924`, and the same object URI and SHA-256.

CPU torch-load command used
`.envs/olcf-rocm711-torch210-py312/bin/python` and
`torch.load(path, map_location="cpu", weights_only=False)`. Exact stdout:

```text
torch_load_path=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt
torch_load_type=dict
torch_load_keys=checkpoint_metadata,loss,model_state_dict,optimizer_state_dict,step
step=1525000
loss=2.4378370428085328
model_state_dict_type=OrderedDict
optimizer_state_dict_type=dict
```

The serialized full-precision loss is consistent with the filename/manifest
loss rounded to four decimals (`2.4378`).

## Commit and terminal evidence reconciliation

The requested source commits are present:

- `9b7783b` canonicalizes the proven job-4962400 launcher.
- `974aae3` makes the renderer compatible with Frontier's default Python.
- `097e6ee` pins the durable job-4962400 evidence paths and tests.
- `40eb8d4` begins the job-4974616 submission evidence.

Their repository integration equivalents are squash commit `42bc407` for the
first three and squash commit `0b654f9` for the complete rerun report. Both are
ancestors of the integration base. The retained rerun report records job
4974616 as `COMPLETED 0:0` across 256 nodes, 2,048 unique rank starts and
accepted updates, finite loss `2.52083`, an advanced generation, a
`5.304643992334604` second merge, and a finalized reload-verified run-local
checkpoint. The proven launcher SHA-256 remains
`106a4dde6b966b0af66a1ac92ea0f459c7a435f81f6e322d92e08f30a2cfad30`.

## Pinning boundary

Both smoke and production launch bundles embed the identical immutable seed
URI, retained absolute path, step, loss, token count, byte size, and SHA-256.
The launch export sets `E97_CHECKPOINT` directly to that retained immutable
file. Neither resolved bundle contains `latest`, and the rendered batch body
remains byte-identical to job 4962400. Only walltime and queue/QoS are profile
parameters.
