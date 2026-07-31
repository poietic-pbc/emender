from __future__ import annotations

import concurrent.futures
import errno
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/frontier/native_g2_artifact_namespace.py"
SUBMIT = ROOT / "scripts/frontier/submit_native_dataplane_2n_gate.sh"
SBATCH = ROOT / "scripts/frontier/native_dataplane_2n_gate.sbatch"
HISTORICAL_JOB_ID = "5109414"


def _module():
    spec = importlib.util.spec_from_file_location(
        "native_g2_artifact_namespace", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _fake_submit_environment(tmp_path: Path, python: str) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    submitted = tmp_path / "submitted"
    rendered = tmp_path / "sbatch-arguments.txt"
    _write_executable(
        fake_bin / "git",
        """\
if [[ " $* " == *" rev-parse HEAD "* ]]; then
  printf '%040d\n' 756
elif [[ " $* " == *" rev-parse --show-toplevel "* ]]; then
  printf '%s\n' "$FAKE_REPO"
elif [[ " $* " == *" branch --show-current "* ]]; then
  printf 'main\n'
elif [[ " $* " == *" diff --quiet "* ]]; then
  exit 0
else
  printf 'unexpected fake git invocation: %s\n' "$*" >&2
  exit 90
fi
""",
    )
    _write_executable(
        fake_bin / "squeue",
        """\
if [[ " $* " == *" -j "* && -e "$FAKE_SUBMITTED" ]]; then
  printf '5109414|PENDING|batch|debug\n'
fi
""",
    )
    _write_executable(
        fake_bin / "scontrol",
        """\
printf '%s\n' \
  'JobId=5109414 JobState=PENDING Account=bif148 QOS=debug' \
  '   Partition=batch NumNodes=2'
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """\
printf '%s\n' "$*" > "$FAKE_RENDERED"
: > "$FAKE_SUBMITTED"
printf '5109414\n'
""",
    )
    _write_executable(
        fake_bin / "ndp-python",
        """\
if [[ ${1:-} == */attest_native_dataplane.py ]]; then
  exit 0
fi
exec "$REAL_PYTHON" "$@"
""",
    )
    artifact_root = tmp_path / "g2"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_REPO": str(ROOT),
            "FAKE_SUBMITTED": str(submitted),
            "FAKE_RENDERED": str(rendered),
            "NDP_ARTIFACT_ROOT": str(artifact_root),
            "NDP_BUILD_MANIFEST": str(tmp_path / "native-artifacts.json"),
            "NDP_PYTHON_BIN": str(fake_bin / "ndp-python"),
            "REAL_PYTHON": python,
            "NDP_RUN_ID": "native-g2-fault-job5109414-regression",
            "NDP_PAYLOAD_ID": "job5109414-regression-payload",
            "NDP_CLEAN_GATE_JSON": str(tmp_path / "clean-gate.json"),
            "REPO": str(ROOT),
            "USER": "artifact-regression",
        }
    )
    Path(env["NDP_BUILD_MANIFEST"]).write_text("{}\n", encoding="utf-8")
    Path(env["NDP_CLEAN_GATE_JSON"]).write_text(
        '{"payload_id":"historical-clean-payload"}\n', encoding="utf-8"
    )
    return env, rendered


def test_job5109414_legacy_order_reproduces_exit_73(tmp_path: Path):
    """The historical monitor created the authoritative batch root first."""
    module = _module()
    root = tmp_path / "g2"
    module.initialize_artifact_root(root)
    legacy = root / HISTORICAL_JOB_ID / "scheduler-evidence"
    legacy.mkdir(parents=True)
    failed = legacy / "terminal.txt"
    failed.write_text("5109414|FAILED|73:0|2|batch|debug\n", encoding="utf-8")

    with pytest.raises(module.ArtifactNamespaceConflict) as error:
        module.publish_batch_namespace(
            root,
            job_id=HISTORICAL_JOB_ID,
            run_id="historical-run",
            payload_id="historical-payload",
        )

    assert error.value.exit_code == 73
    assert failed.read_text(encoding="utf-8") == (
        "5109414|FAILED|73:0|2|batch|debug\n"
    )
    assert not (root / HISTORICAL_JOB_ID / module.OWNER_MARKER).exists()


def test_real_submit_immediate_observation_then_batch_guard_cannot_collide(
    tmp_path: Path,
):
    """Drive the human-equivalent submit/render/observe/batch-guard flow."""
    module = _module()
    env, rendered = _fake_submit_environment(tmp_path, sys.executable)
    submitted = subprocess.run(
        ["bash", str(SUBMIT), "fault"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert submitted.returncode == 0, (
        f"stdout:\n{submitted.stdout}\nstderr:\n{submitted.stderr}"
    )
    root = Path(env["NDP_ARTIFACT_ROOT"])

    assert submitted.stdout.strip() == HISTORICAL_JOB_ID
    ownership = json.loads(
        (root / module.ROOT_SCHEMA_FILE).read_text(encoding="utf-8")
    )
    assert ownership["schema"] == module.OWNERSHIP_SCHEMA
    assert [entry["owner"] for entry in ownership["namespaces"]] == [
        "controller",
        "batch",
        "collector",
    ]
    rendered_text = rendered.read_text(encoding="utf-8")
    assert "-p batch" in rendered_text
    assert "--qos=debug" in rendered_text
    assert "scripts/frontier/native_dataplane_2n_gate.sbatch" in rendered_text
    assert not (root / HISTORICAL_JOB_ID).exists()
    immediate = list(
        (
            root
            / "controller"
            / HISTORICAL_JOB_ID
            / "scheduler-evidence"
        ).glob("immediate-*.json")
    )
    assert len(immediate) == 1
    submitted_records = list(immediate[0].parent.glob("submitted-*.json"))
    assert len(submitted_records) == 1
    assert json.loads(submitted_records[0].read_text())["submission"] == "accepted"
    record = json.loads(immediate[0].read_text(encoding="utf-8"))
    assert record["owner"] == "controller"
    assert record["job_id"] == HISTORICAL_JOB_ID
    assert record["commands"]["squeue"]["stdout"] == (
        "5109414|PENDING|batch|debug\n"
    )
    assert "QOS=debug" in record["commands"]["scontrol"]["stdout"]
    assert "Partition=batch" in record["commands"]["scontrol"]["stdout"]

    batch = module.publish_batch_namespace(
        root,
        job_id=HISTORICAL_JOB_ID,
        run_id=env["NDP_RUN_ID"],
        payload_id=env["NDP_PAYLOAD_ID"],
    )
    assert batch == root / HISTORICAL_JOB_ID
    assert batch.is_symlink()
    marker = json.loads((batch / module.OWNER_MARKER).read_text(encoding="utf-8"))
    assert marker["owner"] == "batch"
    assert marker["job_id"] == HISTORICAL_JOB_ID
    # This represents the first dataplane-setup write after the real guard.
    (batch / "loader-preflight.txt").write_text("dataplane setup entered\n")
    assert (batch / "loader-preflight.txt").is_file()

    payload = SBATCH.read_text(encoding="utf-8")
    assert "native_g2_artifact_namespace.py" in payload
    assert "publish-batch" in payload
    assert 'mkdir -p "$ARTIFACT_DIR"' not in payload


def test_observation_and_collector_reconciliation_are_idempotent_and_disjoint(
    tmp_path: Path,
):
    module = _module()
    root = tmp_path / "g2"
    module.initialize_artifact_root(root)
    failed = {
        "state": "FAILED",
        "exit_code": "73:0",
        "partition": "batch",
        "qos": "debug",
    }
    first = module.record_controller_evidence(
        root, job_id=HISTORICAL_JOB_ID, kind="terminal", evidence=failed
    )
    duplicate = module.record_controller_evidence(
        root, job_id=HISTORICAL_JOB_ID, kind="terminal", evidence=failed
    )
    assert first == duplicate
    assert len(list(first.parent.glob("terminal-*.json"))) == 1

    monitor = module.record_controller_evidence(
        root,
        job_id=HISTORICAL_JOB_ID,
        kind="monitor",
        evidence={"state": "FAILED", "observed_again": True},
    )
    before_batch = module.record_collector_evidence(
        root,
        collector_job_id="5109415",
        payload_job_id=HISTORICAL_JOB_ID,
        kind="registered",
        evidence={"dependency": "afterany:5109414"},
    )
    batch = module.publish_batch_namespace(
        root,
        job_id=HISTORICAL_JOB_ID,
        run_id="run",
        payload_id="payload",
    )
    after_batch = module.record_collector_evidence(
        root,
        collector_job_id="5109415",
        payload_job_id=HISTORICAL_JOB_ID,
        kind="terminal",
        evidence={"payload_state": "FAILED", "exit_code": "73:0"},
    )

    assert first.is_relative_to(root / "controller")
    assert monitor.is_relative_to(root / "controller")
    assert before_batch.is_relative_to(root / "collectors" / "5109415")
    assert after_batch.is_relative_to(root / "collectors" / "5109415")
    assert not first.is_relative_to(batch)
    assert not before_batch.is_relative_to(batch)
    assert json.loads((batch / module.OWNER_MARKER).read_text())["owner"] == "batch"


def test_conflicting_authoritative_batch_artifacts_fail_closed_without_overwrite(
    tmp_path: Path,
):
    module = _module()
    root = tmp_path / "g2"
    module.initialize_artifact_root(root)
    conflict = root / "5109500"
    conflict.mkdir()
    retained = conflict / "failed.txt"
    retained.write_text("prior authoritative evidence\n", encoding="utf-8")

    with pytest.raises(module.ArtifactNamespaceConflict) as error:
        module.publish_batch_namespace(
            root, job_id="5109500", run_id="new", payload_id="new"
        )

    assert error.value.exit_code == 73
    assert retained.read_text(encoding="utf-8") == "prior authoritative evidence\n"
    assert list(conflict.iterdir()) == [retained]


@pytest.mark.parametrize("owner_root", ["controller", "collectors"])
def test_evidence_writers_reject_cross_owner_symlink(owner_root: str, tmp_path: Path):
    module = _module()
    root = tmp_path / "g2"
    module.initialize_artifact_root(root)
    batch = root / "5111500"
    batch.mkdir()
    retained = batch / "retained.txt"
    retained.write_text("batch-owned\n", encoding="utf-8")
    (root / owner_root).symlink_to(batch, target_is_directory=True)

    with pytest.raises(module.ArtifactNamespaceConflict):
        if owner_root == "controller":
            module.record_controller_evidence(
                root,
                job_id="5111500",
                kind="monitor",
                evidence={"state": "RUNNING"},
            )
        else:
            module.record_collector_evidence(
                root,
                collector_job_id="5111501",
                payload_job_id="5111500",
                kind="terminal",
                evidence={"state": "FAILED"},
            )

    assert list(batch.iterdir()) == [retained]
    assert retained.read_text(encoding="utf-8") == "batch-owned\n"


def test_batch_publication_is_single_winner_under_duplicate_and_mkdir_races(
    tmp_path: Path,
):
    module = _module()
    root = tmp_path / "g2"
    module.initialize_artifact_root(root)

    barrier = threading.Barrier(2)

    def publish():
        barrier.wait()
        try:
            return module.publish_batch_namespace(
                root, job_id="5109600", run_id="run", payload_id="payload"
            )
        except module.ArtifactNamespaceConflict as error:
            return error

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = [executor.submit(publish) for _ in range(2)]
    values = [future.result() for future in results]
    assert sum(isinstance(value, Path) for value in values) == 1
    assert sum(
        isinstance(value, module.ArtifactNamespaceConflict) for value in values
    ) == 1
    assert json.loads(
        (root / "5109600" / module.OWNER_MARKER).read_text(encoding="utf-8")
    )["owner"] == "batch"

    for offset in range(12):
        job_id = str(5109700 + offset)
        race = threading.Barrier(2)

        def publish_again():
            race.wait()
            try:
                module.publish_batch_namespace(
                    root, job_id=job_id, run_id="run", payload_id="payload"
                )
                return "batch"
            except module.ArtifactNamespaceConflict:
                return "conflict"

        def mkdir_again():
            race.wait()
            try:
                (root / job_id).mkdir()
                return "mkdir"
            except FileExistsError:
                return "exists"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            batch_future = executor.submit(publish_again)
            mkdir_future = executor.submit(mkdir_again)
        outcomes = {batch_future.result(), mkdir_future.result()}
        assert outcomes in ({"batch", "exists"}, {"conflict", "mkdir"})
        final = root / job_id
        if "batch" in outcomes:
            assert (final / module.OWNER_MARKER).is_file()
        else:
            assert list(final.iterdir()) == []


def test_batch_publication_cannot_be_replaced_by_concurrent_directory_rename(
    tmp_path: Path,
):
    module = _module()
    root = tmp_path / "g2"
    module.initialize_artifact_root(root)

    for offset in range(24):
        job_id = str(5110000 + offset)
        external = root / f".external-{job_id}"
        external.mkdir()
        external_marker = external / "external.txt"
        external_marker.write_text("external conflict\n", encoding="utf-8")
        race = threading.Barrier(2)

        def publish():
            race.wait()
            try:
                module.publish_batch_namespace(
                    root, job_id=job_id, run_id="run", payload_id="payload"
                )
                return "batch"
            except module.ArtifactNamespaceConflict:
                return "conflict"

        def rename():
            race.wait()
            try:
                os.rename(external, root / job_id)
                return "renamed"
            except OSError as error:
                assert error.errno in {
                    errno.EEXIST,
                    errno.ENOTEMPTY,
                    errno.EACCES,
                    errno.ENOTDIR,
                    errno.EISDIR,
                }
                return "blocked"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            batch_future = executor.submit(publish)
            rename_future = executor.submit(rename)
        outcomes = {batch_future.result(), rename_future.result()}
        assert outcomes in ({"batch", "blocked"}, {"conflict", "renamed"})
        final = root / job_id
        if "batch" in outcomes:
            assert (final / module.OWNER_MARKER).is_file()
            assert not (final / "external.txt").exists()
        else:
            assert (final / "external.txt").read_text(encoding="utf-8") == (
                "external conflict\n"
            )
            assert not (final / module.OWNER_MARKER).exists()


def test_cli_returns_73_for_an_existing_authoritative_root(tmp_path: Path):
    root = tmp_path / "g2"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "init-root",
            "--artifact-root",
            str(root),
        ],
        check=True,
    )
    (root / "5111000").mkdir()
    conflict = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "publish-batch",
            "--artifact-root",
            str(root),
            "--job-id",
            "5111000",
            "--run-id",
            "run",
            "--payload-id",
            "payload",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert conflict.returncode == 73
    assert "refusing to overwrite retained job evidence" in conflict.stderr
