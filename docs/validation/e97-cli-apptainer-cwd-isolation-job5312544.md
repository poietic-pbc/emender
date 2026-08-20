# E97 CLI cwd-only Apptainer isolation qualification

Job: 5312544  
Authority: `/lustre/orion/bif148/proj-shared/emender/evaluations/e97-cli-apptainer-isolation-v1`

## Verdict

Apptainer is the best stock isolation mechanism currently installed on Frontier for the proposed discoverable CLI agent. A carefully configured Apptainer invocation successfully exposed exactly one writable task directory while hiding Frontier project, home, and software paths; disabling network access; removing the host environment; making the image root read-only; and hiding `/proc` and `/sys`.

All eight Frontier ranks passed the isolation probe in job 5312544.

This qualifies the filesystem and network boundary needed to ensure that commands such as `rm` and `mv` can affect only the disposable task working directory. It is not yet a complete hostile-code sandbox: Apptainer shares the host kernel, no seccomp profile was applied, and per-command cgroup resource limits remain unqualified.

## Installed-system survey

Available:

- Apptainer 1.4.5;
- Singularity compatibility command;
- Podman;
- `unshare`;
- `systemd-run`, `prlimit`, `timeout`, and `setpriv`.

Not installed:

- bubblewrap;
- nsjail;
- firejail;
- Docker.

Rootless Podman reported no `/etc/subuid` allocation for the user and selected storage under the network-mounted home filesystem. Podman documentation cautions that rootless container storage on NFS/Lustre/GPFS is problematic. It is therefore not the preferred Frontier path without additional site configuration and node-local storage setup.

Unprivileged namespaces are available through `unshare`, but using them directly would require implementing and maintaining our own mount, device, network, environment, and process policy. Apptainer already provides these mechanisms and is supported by OLCF.

The login-node kernel returned `ENOSYS` for the Landlock ruleset syscall despite shipping Landlock headers, so Landlock is not currently available as a second filesystem enforcement layer.

## Important unsafe default

Frontier's Apptainer configuration automatically binds paths including:

- `/lustre/orion`;
- `/autofs/nccs-svm1_home*`;
- `/ccs/home`;
- `/ccs/sw`;
- `/sw/frontier`.

Using ordinary `apptainer exec` would therefore expose sensitive host/project paths. `--containall` alone is insufficient as an authority. The promoted invocation must disable administrator bind paths explicitly.

## Qualified invocation

The probe used:

```bash
apptainer exec \
  --containall \
  --cleanenv \
  --net --network none \
  --no-privs --drop-caps all \
  --no-mount bind-paths,home,cwd,tmp,hostfs,proc,sys \
  --bind "$SANDBOX:/work:rw" \
  --cwd /work \
  IMAGE.sif \
  COMMAND ARGS...
```

The outer launcher additionally used:

```bash
timeout --signal=TERM --kill-after=5 60 ...
```

The sandbox path is canonicalized by the launcher and is intended to be a disposable task directory. The SIF image is immutable and hash-verified before execution.

## Checks performed on every rank

Each rank verified that:

1. `/work` was the current directory.
2. A host-created file inside the task directory was readable.
3. A result written inside `/work` appeared in the host task directory.
4. `/lustre`, `/autofs`, and `/ccs` did not exist inside the container.
5. `/proc/self/status` and `/sys/kernel` were unavailable.
6. a host secret environment variable was absent.
7. HTTPS access to `example.com` failed under the loopback-only network namespace.
8. writing `/outside.txt` failed because the image root was read-only.
9. a sibling host sentinel remained unchanged.

Machine summary:

```json
{
  "checks": {
    "clean_environment": true,
    "host_paths_hidden": true,
    "network_isolated": true,
    "proc_sys_hidden": true,
    "root_read_only": true
  },
  "ranks": 8,
  "schema": "emender-e97-cli-apptainer-isolation-v1",
  "status": "pass"
}
```

## Image authority

Qualification used a minimal Alpine 3.20 SIF:

`/lustre/orion/bif148/proj-shared/emender/agent_sandbox/images/alpine-3.20.sif`

SHA-256:

`ff23e743832f57c91c9c9c0f3e1296cde5d8c8a9299f9a751247e63a318a00f0`

This image is sufficient for isolation qualification but not for the agent CLI. A production image must be built immutably with a small reviewed tool set such as Git, ripgrep, jq, Python, and the discoverable repository CLI.

## Scheduler evidence

Job 5312544:

- `Partition=batch`;
- `QOS=debug`;
- `Requeue=0`;
- `COMPLETED`;
- exit `0:0`;
- elapsed `00:07:16`;
- node `frontier03202`.

Accounting artifact:

`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-cli-apptainer-isolation-v1/identity/sacct-5312544.txt`

## Remaining gaps

### Resource exhaustion

Apptainer's `--memory`, `--cpus`, and `--pids-limit` options failed in the login environment because the expected delegated cgroup path was unavailable. Job-level Slurm cgroups and the outer wall-clock timeout provide partial containment, but a malicious fork or memory bomb needs a qualified per-command limit.

Candidate next controls:

1. qualify Apptainer cgroup options inside a Slurm job;
2. otherwise run each command in a bounded Slurm step/cgroup;
3. use `systemd-run --user --scope` where a user systemd manager is available;
4. apply `prlimit` for CPU time, file size, open files, address space, and core dumps;
5. retain process-group termination and a hard wall timeout.

`RLIMIT_NPROC` must be used cautiously because limits are charged to the shared host UID rather than only the sandbox unless a user/PID namespace changes that accounting.

### Kernel attack surface

Apptainer shares the host kernel. Removing `/proc` and `/sys`, dropping capabilities, setting `NoNewPrivs`, and using an isolated network substantially reduce exposure but do not equal a VM, gVisor, or Firecracker boundary. No such stronger stock runtime was found installed on Frontier.

A reviewed seccomp profile would provide useful defense in depth if it can be applied in the site configuration without breaking required CLI tools.

### Operational policy

The CLI tool must pass an argument vector directly and must not use a host shell. A shell may exist inside the isolated container, but it has only the disposable cwd, immutable image, empty network namespace, scrubbed environment, and bounded lifetime.

The Pi process itself remains outside the container and must never expose its host environment, session files, checkpoint paths, or credentials to the command.

## Decision

Use Apptainer as the initial discoverable CLI sandbox on Frontier, with the exact fail-closed mount and network options above. Do not use Apptainer defaults. Do not expose a CLI tool until the production image, output limits, process/resource limits, command transaction behavior, and injection/cycle tests are also qualified.

References:

- OLCF, Containers on Frontier: `https://docs.olcf.ornl.gov/software/containers_on_frontier.html`
- Apptainer bind and mount control: `https://apptainer.org/docs/user/1.4/bind_paths_and_mounts.html`
- Apptainer network namespaces: `https://apptainer.org/docs/user/1.4/networking.html`
- Apptainer security options: `https://apptainer.org/docs/user/1.4/security_options.html`
