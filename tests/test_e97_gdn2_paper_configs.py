import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/frontier/e97_gdn2_paper"


def _load(name):
    return json.loads((CONFIG_ROOT / name).read_text())


def test_primary_arm_configs_are_matched_and_exact():
    nonlinear = _load("e97_mlp.json")
    linear = _load("e97_linear_mlp.json")
    differences = {
        key for key in set(nonlinear) | set(linear)
        if nonlinear.get(key) != linear.get(key)
    }
    assert differences == {"arm", "linear_state", "use_chunked_e97"}
    assert nonlinear["linear_state"] == 0
    assert linear["linear_state"] == 1
    assert nonlinear["exact_parameters"] == linear["exact_parameters"] == 1_286_589_072
    assert nonlinear["derived_mlp_hidden"] == 4032


def test_gdn2_config_is_size_matched_and_source_bound():
    e97 = _load("e97_mlp.json")
    gdn2 = _load("gdn2_mlp.json")
    assert gdn2["exact_parameters"] == 1_285_245_320
    assert abs(e97["exact_parameters"] - gdn2["exact_parameters"]) / gdn2[
        "exact_parameters"] < 0.0011
    assert gdn2["external_gdn2_commit"] == "95709fc250357c2dd109361c353192f2aa5913f9"


def test_train_argv_uses_exact_steps_without_zero_minute_budget():
    output = subprocess.check_output([
        sys.executable,
        str(ROOT / "scripts/render_e97_gdn2_paper_args.py"),
        "--arm", "e97-linear-mlp", "--world-size", "8",
        "--steps", "160", "--data", "/data", "--output", "/output",
    ], cwd=ROOT)
    argv = [item.decode() for item in output.split(b"\0") if item]
    assert argv[argv.index("--steps") + 1] == "160"
    assert "--train_minutes" not in argv


def test_manifest_renderer_reproduces_committed_manifests(tmp_path):
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/validate_e97_gdn2_paper_configs.py"),
        "--gdn2-path", str(ROOT / "src/GatedDeltaNet-2"),
        "--output", str(tmp_path),
    ], check=True, cwd=ROOT)
    for arm in ("e97-mlp", "e97-linear-mlp", "gdn2-mlp"):
        expected = json.loads((CONFIG_ROOT / "manifests" / f"{arm}.json").read_text())
        observed = json.loads((tmp_path / f"{arm}.json").read_text())
        assert observed == expected
