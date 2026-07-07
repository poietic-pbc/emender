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
    assert "frontier_require_requested_rccl_net_plugin()" in helper
    assert '"/rocm/*/lib/librccl-net.so' in helper
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


def test_e97_ladder_forces_olcf_runtime_and_plugin_gate():
    ladder = (ROOT / "scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch").read_text()
    canary = (ROOT / "scripts/frontier/e97_1p3b_pretrained_canary.sbatch").read_text()

    assert ".envs/olcf-rocm711-torch210-py312" in ladder
    assert "FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-1}" in ladder
    assert "FRONTIER_RCCL_NET_PLUGIN_MODULE=${FRONTIER_RCCL_NET_PLUGIN_MODULE:-rccl-net-plugin/1.0}" in ladder
    assert "REQUIRE_RCCL_NET_PLUGIN=${REQUIRE_RCCL_NET_PLUGIN:-1}" in ladder
    assert "NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=${NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS:-1800}" in ladder
    assert "frontier_load_default_modules" in canary
    assert "frontier_derive_master_port" in canary
    assert 'RCCL_NET_PLUGIN_STATUS=$(frontier_require_requested_rccl_net_plugin "runtime setup")' in canary
    assert 'frontier_require_requested_rccl_net_plugin "delegated srun preflight"' in canary
    assert 'frontier_require_requested_rccl_net_plugin "delegated srun rank ${SLURM_PROCID:-unknown}"' in canary
    assert "--distributed_init_timeout_seconds" in canary
    assert "RANK_START_LOG" in canary


def test_trainpy_step1065000_smoke_requires_olcf_rccl_plugin():
    smoke = (ROOT / "scripts/frontier/e97_1p3b_step1065000_b4_trainpy_smoke.sbatch").read_text()

    assert "BATCH_SIZE=${BATCH_SIZE:-4}" in smoke
    assert "DILOCO_K=${DILOCO_K:-40}" in smoke
    assert "DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-1}" in smoke
    assert "FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-1}" in smoke
    assert "FRONTIER_RUNTIME_PROFILE=${FRONTIER_RUNTIME_PROFILE:-olcf-rccl-debug}" in smoke
    assert "FRONTIER_RCCL_NET_PLUGIN_MODULE=${FRONTIER_RCCL_NET_PLUGIN_MODULE:-rccl-net-plugin/1.0}" in smoke
    assert "REQUIRE_RCCL_NET_PLUGIN=${REQUIRE_RCCL_NET_PLUGIN:-1}" in smoke
    assert "FRONTIER_TRAIN_ENV_PREFLIGHT=${FRONTIER_TRAIN_ENV_PREFLIGHT:-1}" in smoke


def test_e97_canary_reasserts_emender_conda_env_inside_srun_ranks():
    helper = (ROOT / "scripts/frontier/frontier_runtime_env.sh").read_text()
    canary = (ROOT / "scripts/frontier/e97_1p3b_pretrained_canary.sbatch").read_text()

    assert "frontier_activate_emender_conda_env()" in helper
    assert "frontier_assert_emender_conda_env()" in helper
    assert "conda deactivate" not in helper
    assert "conda deactivate" not in canary
    assert "frontier_activate_emender_conda_env" in canary
    assert "export EMENDER_CONDA_ENV" in canary
    assert "frontier_train_env_preflight=$FRONTIER_TRAIN_ENV_PREFLIGHT" in canary
    assert "=== delegated srun training env preflight ===" in canary
    assert "frontier_capture_runtime_env" in canary

    preflight = canary.index("=== delegated srun training env preflight ===")
    training = canary.index("2>&1 | tee \"${LOG_DIR}/train.log\"")
    for section in (canary[preflight:training], canary[training - 800:training]):
        activation = section.index("frontier_activate_emender_conda_env")
        python_exec = section.find("python -u train.py")
        if python_exec != -1:
            assert activation < python_exec


def test_updated_olcf_debug_wrapper_preserves_chain_guard_and_hardening():
    debug = (ROOT / "scripts/frontier/e97_updated_olcf_runtime_debug.sbatch").read_text()

    assert "ENV_PREFIX=${ENV_PREFIX:-\"${REPO}/.envs/olcf-rocm711-torch210-py312\"}" in debug
    assert "module load rccl-net-plugin/1.0" in debug
    assert "RESOLVED_PRODUCTION_LATEST=$(readlink -f \"${PRODUCTION_LATEST}\")" in debug
    assert "export CHAIN_LATEST_PATH=" in debug
    assert "export CHAIN_MANIFEST_PATH=" in debug
    assert "export CHAIN_UPDATE_ON_FAILURE=0" in debug
    assert "export FRONTIER_TRAIN_ENV_PREFLIGHT=${FRONTIER_TRAIN_ENV_PREFLIGHT:-1}" in debug
    assert "frontier_activate_emender_conda_env" in debug
    assert "source activate" not in debug
    assert "conda deactivate" not in debug
    assert "FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-1}" in debug
    assert "REQUIRE_RCCL_NET_PLUGIN=${REQUIRE_RCCL_NET_PLUGIN:-1}" in debug
    assert "NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=${NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS:-1800}" in debug
    assert "production latest.pt metadata changed during debug smoke" in debug


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


def test_train_runtime_manifest_resolves_olcf_rocm_scoped_librccl_net(monkeypatch, tmp_path):
    import train

    lib_dir = tmp_path / "rocm" / "7.1.1" / "lib"
    lib_dir.mkdir(parents=True)
    plugin = lib_dir / "librccl-net.so"
    plugin.write_text("placeholder")

    monkeypatch.setenv("OLCF_OFI_NCCL_ROOT", str(tmp_path))
    monkeypatch.delenv("FRONTIER_ROCM_MODULE", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    manifest = train.frontier_runtime_manifest()

    assert manifest["librccl_net_path"] == str(plugin)
