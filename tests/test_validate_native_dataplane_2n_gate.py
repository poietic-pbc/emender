from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/frontier/validate_native_dataplane_2n_gate.py"
SBATCH = ROOT / "scripts/frontier/native_dataplane_2n_gate.sbatch"
SUBMIT = ROOT / "scripts/frontier/submit_native_dataplane_2n_gate.sh"
FABRIC = ROOT / "native/dataplane/src/fabric.cpp"
RUNNER = ROOT / "native/dataplane/src/frontier_2n_gate.cpp"
PROTOCOL = ROOT / "native/dataplane/src/protocol.cpp"
JOB_5031115 = ROOT / "reports/frontier/native-dataplane/5031115"


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
    assert "#SBATCH --network=job_vni" in payload
    assert "--network=job_vni" in submit
    assert "--network=single_node_vni" not in payload
    assert "FI_PROVIDER=cxi" in submit
    assert "--nodes=2 --ntasks=2 --ntasks-per-node=1" in payload
    assert "NDP_CLEAN_GATE_JSON" in submit
    assert "fault payload ID is unchanged" in submit
    forbidden = ("mpirun", "MPI_Allreduce", "torchrun", "gpus-per-node")
    assert all(term not in payload for term in forbidden)


def test_native_cxi_setup_uses_provider_mr_contract():
    source = FABRIC.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "hints->caps = FI_MSG | FI_SOURCE;" in source
    assert "info_->domain_attr->mr_mode & FI_MR_ENDPOINT" in source
    assert "fi_mr_bind(slot->mr, &endpoint_->fid, 0)" in source
    assert "fi_mr_enable(slot->mr)" in source
    assert 'const std::string domain{"cxi0"};' in runner
    assert "config.domain_len" in runner


def test_native_gate_fences_contribution_before_redistribution():
    source = RUNNER.read_text(encoding="utf-8")
    assert "MessageType::result_announce" in source
    assert "peer_result_announced_" in source
    assert "send_result_announce" in source
    assert "result announce did not match the frozen generation" in source
    assert "MessageType::goodbye" in source
    assert "peer_redistribution_complete" in source
    assert "send_generation_goodbye" in source
    assert "generation goodbye did not match the frozen generation" in source


def test_exact_validator_accepts_rank_specific_admission_below_bound():
    module = _module()
    for rank in (0, 1):
        node = json.loads((JOB_5031115 / f"node-{rank}.json").read_text(encoding="utf-8"))
        module._validate_common_node(node, rank=rank, mode="clean", provider="cxi", exact=True)


def test_native_full_layout_uses_bounded_dense_pipeline_and_cached_validation():
    runner = RUNNER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    fabric = FABRIC.read_text(encoding="utf-8")
    assert "active_count_ < kDenseWindow" in runner
    assert "constexpr std::size_t kDenseWindow = kSlots - 1" in runner
    assert "pending_fetches" in runner
    assert "pending_frames_" in runner
    assert "poll_receive" in runner
    assert "pending_frames_.push_back" in runner
    assert "encode_frame_prehashed" in runner
    assert "payload_validated" in protocol
    assert "event.detail = sha256" not in fabric
