from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_cli_apptainer_isolation_1n.sbatch"
PROBE = ROOT / "scripts/probe_e97_cli_apptainer_isolation.sh"


def test_frontier_isolation_launcher_is_fixed_world_and_fail_closed():
    text = LAUNCHER.read_text()
    for required in (
        "#SBATCH -p batch",
        "#SBATCH -q debug",
        "#SBATCH --ntasks-per-node=8",
        "#SBATCH --no-requeue",
        "Partition=batch",
        "QOS=debug",
        "Requeue=0",
        "sha256sum -c -",
        "git archive",
        "--kill-on-bad-exit=1",
    ):
        assert required in text


def test_probe_exposes_only_cwd_and_removes_network_host_mounts_and_environment():
    text = PROBE.read_text()
    for required in (
        "--containall",
        "--cleanenv",
        "--net --network none",
        "--no-privs --drop-caps all",
        "--no-mount bind-paths,home,cwd,tmp,hostfs,proc,sys",
        '--bind "$SANDBOX:/work:rw"',
        "--cwd /work",
        "test ! -e /lustre",
        "test ! -e /autofs",
        "test ! -e /ccs",
        "test ! -e /proc/self/status",
        "test ! -e /sys/kernel",
        "EMENDER_HOST_SECRET",
        "https://example.com",
        "/outside.txt",
        "timeout --signal=TERM --kill-after=5 60",
    ):
        assert required in text
