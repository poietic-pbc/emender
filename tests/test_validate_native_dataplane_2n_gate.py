from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/frontier/validate_native_dataplane_2n_gate.py"
SBATCH = ROOT / "scripts/frontier/native_dataplane_2n_gate.sbatch"
SUBMIT = ROOT / "scripts/frontier/submit_native_dataplane_2n_gate.sh"
FABRIC = ROOT / "native/dataplane/src/fabric.cpp"


def _module():
    spec = importlib.util.spec_from_file_location("validate_native_dataplane_2n_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_exact_reference_matches_retained_e97_shape():
    module = _module()
    assert module.REQUIRED_SPEEDUP_OVER_PYTHON == 4.0
    assert module.NATIVE_TARGET_SECONDS == module.PYTHON_BASELINE_SECONDS / 4.0
    assert module.MIN_LOGICAL_BYTES_PER_SECOND == (
        4.0 * module.PYTHON_BASELINE_LOGICAL_BYTES_PER_SECOND
    )
    reference = module.exact_reference(module.LAYOUT_BYTES, module.PAYLOAD_MAX)
    assert len(reference["shard_sha256"]) == 83
    assert reference["even_value"] != reference["odd_value"]
    assert len(reference["payload_sha256"]) == 64
    root = module.expected_result_root(
        run_id="run", payload_id="payload", generation=0, owner_epoch=1,
        layout_bytes=module.LAYOUT_BYTES, payload_max=module.PAYLOAD_MAX,
        shard_sha256=reference["shard_sha256"],
    )
    assert len(root) == 64


def test_reference_rejects_partial_f64_pair_layout():
    module = _module()
    try:
        module.exact_reference(24, 16)
    except ValueError as error:
        assert "pairs" in str(error)
    else:
        raise AssertionError("partial alternating f64 reference was accepted")


def test_frontier_payload_is_exactly_two_node_debug_cxi_and_fault_is_gated():
    payload = SBATCH.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "#SBATCH -q debug" in payload
    assert "#SBATCH -N 2" in payload
    assert "#SBATCH -t 00:20:00" in payload
    assert "#SBATCH --network=single_node_vni,job_vni" in payload
    assert "--network=single_node_vni,job_vni" in submit
    assert payload.count("--network=single_node_vni,job_vni") == 1
    assert "FI_PROVIDER=cxi" in submit
    assert "--nodes=2 --ntasks=2 --ntasks-per-node=1" in payload
    assert "NDP_CLEAN_GATE_JSON" in submit
    assert "fault payload ID is unchanged" in submit
    forbidden = ("mpirun", "MPI_Allreduce", "torchrun", "gpus-per-node")
    assert all(term not in payload for term in forbidden)


def test_native_cxi_setup_uses_provider_mr_contract():
    source = FABRIC.read_text(encoding="utf-8")
    assert "hints->caps = FI_MSG;" in source
    assert "if (!config_.production) hints->caps |= FI_SOURCE;" in source
    assert "info_->domain_attr->mr_mode & FI_MR_ENDPOINT" in source
    assert "fi_mr_bind(slot->mr, &endpoint_->fid, 0)" in source
    assert "fi_mr_enable(slot->mr)" in source
