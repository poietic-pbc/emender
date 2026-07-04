from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontier_runtime_helper_loads_olcf_plugin_after_rocm():
    helper = (ROOT / "scripts/frontier/frontier_runtime_env.sh").read_text()

    rocm_load = helper.index('module load "$rocm_module"')
    plugin_load = helper.index('module load "$rccl_plugin_module"')
    assert rocm_load < plugin_load
    assert "FRONTIER_ENABLE_OLCF_RCCL_PLUGIN" in helper
    assert "FRONTIER_RUNTIME_PROFILE" in helper
    assert "rccl-net-plugin/1.0" in helper
    assert 'export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-librccl-net.so}"' in helper
    assert "OLCF_OFI_NCCL_ROOT" in helper
    assert "librccl_net_path" in helper
    assert "torch.version.hip" in helper
    assert "triton.__version__" in helper
    assert "python_version" in helper


def test_debug_smoke_uses_frontier_runtime_capture():
    smoke = (ROOT / "scripts/frontier/debug_smoke_one_node.slurm").read_text()

    assert "frontier_runtime_env.sh" in smoke
    assert "frontier_load_default_modules" in smoke
    assert "LIBRCCL_NET_PATH=$(frontier_resolve_librccl_net)" in smoke
    assert "frontier_capture_runtime_env" in smoke
    assert '"librccl_net_path": "${LIBRCCL_NET_PATH}"' in smoke
    assert '"olcf_ofi_nccl_root": "${OLCF_OFI_NCCL_ROOT:-}"' in smoke
    assert '"nccl_net_plugin": "${NCCL_NET_PLUGIN:-}"' in smoke


def test_rccl_diagnostic_uses_shared_plugin_resolution():
    diag = (ROOT / "scripts/frontier/rccl_allreduce_diag.sbatch").read_text()

    assert "frontier_runtime_env.sh" in diag
    assert "frontier_load_default_modules" in diag
    assert "PLUGIN_STATUS=$(frontier_resolve_librccl_net)" in diag
    assert "frontier_capture_runtime_env" in diag


def test_train_runtime_manifest_resolves_olcf_librccl_net(monkeypatch, tmp_path):
    import train

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    plugin = lib_dir / "librccl-net.so"
    plugin.write_text("placeholder")

    monkeypatch.setenv("OLCF_OFI_NCCL_ROOT", str(tmp_path))
    monkeypatch.setenv("NCCL_NET_PLUGIN", "librccl-net.so")
    monkeypatch.setenv("FI_CXI_RX_MATCH_MODE", "hybrid")
    monkeypatch.setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1")

    manifest = train.frontier_runtime_manifest()

    assert manifest["olcf_ofi_nccl_root"] == str(tmp_path)
    assert manifest["librccl_net_path"] == str(plugin)
    assert manifest["env"]["NCCL_NET_PLUGIN"] == "librccl-net.so"
    assert manifest["env"]["FI_CXI_RX_MATCH_MODE"] == "hybrid"
    assert manifest["env"]["HSA_FORCE_FINE_GRAIN_PCIE"] == "1"
    assert manifest["torch_version"]
    assert "python_version" in manifest
    assert "triton_version" in manifest
