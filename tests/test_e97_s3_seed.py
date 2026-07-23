import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts.frontier import materialize_e97_s3_seed as seed_materializer
from scripts.frontier.materialize_e97_s3_seed import materialize, verify_authorities


SEED = json.loads((Path(__file__).parents[1] / "configs/frontier/e97_async_256.yaml").read_text())["seed"]


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def opener(objects):
    def open_url(url, timeout):
        if url not in objects:
            raise OSError("inaccessible object")
        return Response(objects[url])
    return open_url


def authorities(checkpoint=b"checkpoint", **changes):
    seed = {**SEED, "size": len(checkpoint), "sha256": hashlib.sha256(checkpoint).hexdigest()}
    manifest = {"checkpoint_s3_uri": seed["uri"], "checkpoint_size_bytes": seed["size"],
                "checkpoint_sha256": seed["sha256"], "step": seed["step"], "loss": seed["loss"]}
    latest = dict(manifest)
    target = changes.pop("target", None)
    if target:
        (manifest if target == "manifest" else latest).update(changes)
    from scripts.frontier.materialize_e97_s3_seed import https_url
    objects = {https_url(seed["manifest_uri"]): json.dumps(manifest).encode(),
               https_url(seed["latest_pointer_uri"]): json.dumps(latest).encode(),
               https_url(seed["uri"]): checkpoint}
    return seed, objects


@pytest.mark.parametrize("target,change", [
    ("latest", {"checkpoint_s3_uri": "s3://bucket/drift.pt"}),
    ("manifest", {"checkpoint_size_bytes": 1}),
    ("manifest", {"checkpoint_sha256": "0" * 64}),
    ("latest", {"step": 1}),
    ("latest", {"loss": 1.0}),
])
def test_authority_disagreement_fails_closed(target, change):
    seed, objects = authorities(target=target, **change)
    with pytest.raises(ValueError, match="disagrees"):
        verify_authorities(seed, opener(objects))


def test_absent_field_and_inaccessible_source_fail_closed():
    seed, objects = authorities()
    document_url = next(iter(objects))
    value = json.loads(objects[document_url]); value.pop("step")
    objects[document_url] = json.dumps(value).encode()
    with pytest.raises(ValueError, match="disagrees"):
        verify_authorities(seed, opener(objects))
    with pytest.raises(OSError, match="inaccessible"):
        verify_authorities(seed, opener({}))


def test_atomic_job_scoped_materialization_and_runtime_identity(tmp_path, monkeypatch):
    seed, objects = authorities()
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    destination = tmp_path / "emender-e97-seed-1234" / "checkpoint.pt"
    runtime = tmp_path / "artifacts" / "seed-materialization.json"
    assert materialize(seed, destination, runtime, opener(objects)) == destination
    assert destination.read_bytes() == b"checkpoint"
    evidence = json.loads(runtime.read_text())
    assert evidence["staged_size"] == seed["size"]
    assert evidence["staged_sha256"] == seed["sha256"]
    assert not list(destination.parent.glob("*.tmp.*"))


@pytest.mark.parametrize("download", [
    b"partial",
    b"corrupt!",
])
def test_wrong_size_or_hash_is_removed_and_never_promoted(
        tmp_path, monkeypatch, download):
    seed, objects = authorities(checkpoint=b"complete")
    from scripts.frontier.materialize_e97_s3_seed import https_url
    objects[https_url(seed["uri"])] = download
    monkeypatch.setenv("SLURM_JOB_ID", "5678")
    destination = tmp_path / "emender-e97-seed-5678" / "checkpoint.pt"
    with pytest.raises(ValueError, match="identity mismatch"):
        materialize(seed, destination, tmp_path / "runtime.json", opener(objects))
    assert not destination.exists()
    assert not list(destination.parent.glob("*.tmp.*"))


def test_stale_file_non_job_path_and_legacy_path_are_rejected(tmp_path, monkeypatch):
    seed, objects = authorities()
    monkeypatch.setenv("SLURM_JOB_ID", "9012")
    destination = tmp_path / "emender-e97-seed-9012" / "checkpoint.pt"
    destination.parent.mkdir(); destination.write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="stale"):
        materialize(seed, destination, tmp_path / "runtime.json", opener(objects))
    legacy = tmp_path / "legacy-shared-seed" / "checkpoint.pt"
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        materialize(seed, legacy, tmp_path / "runtime.json", opener(objects))
    assert not legacy.parent.exists()


def test_shared_filesystem_seed_destination_is_rejected_before_download(
        tmp_path, monkeypatch):
    seed, _ = authorities()
    monkeypatch.setenv("SLURM_JOB_ID", "3456")
    monkeypatch.setattr(
        seed_materializer, "_filesystem_type", lambda _path: "lustre")
    destination = tmp_path / "emender-e97-seed-3456" / "checkpoint.pt"
    with pytest.raises(ValueError, match="shared filesystem"):
        materialize(
            seed, destination, tmp_path / "runtime.json",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("shared destination must fail before S3 access")))
    assert not destination.exists()


@pytest.mark.parametrize("job_id,directory", [
    (None, "emender-e97-seed-5059548"),
    ("", "emender-e97-seed-5059548"),
    ("5059548", "emender-e97-seed-5059549"),
    ("5059548", "5059548"),
])
def test_destination_job_scope_fails_closed(
        tmp_path, monkeypatch, job_id, directory):
    if job_id is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
    with pytest.raises(ValueError, match="current SLURM_JOB_ID"):
        seed_materializer._validate_destination(
            tmp_path / directory / "checkpoint.pt")


def test_seed_config_loader_rejects_noncanonical_shape(tmp_path):
    bad = tmp_path / "config.json"
    bad.write_text(json.dumps({"seed": {**SEED, "extra": True}}))
    with pytest.raises(ValueError, match="unknown or missing seed fields"):
        seed_materializer.load_seed_config(bad)
