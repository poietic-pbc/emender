import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs/pi/e97-dense-agent.models.json"
TOOLS = ROOT / "configs/pi/e97-v1-tools.ts"


def test_pi_model_config_is_bounded_state_affine_openai_chat():
    config = json.loads(MODEL_CONFIG.read_text())
    provider = config["providers"]["emender-local"]
    model = provider["models"][0]
    assert provider["baseUrl"] == "http://127.0.0.1:8797/v1"
    assert provider["api"] == "openai-completions"
    assert provider["compat"]["supportsDeveloperRole"] is False
    assert provider["compat"]["supportsUsageInStreaming"] is False
    assert provider["compat"]["sendSessionAffinityHeaders"] is True
    assert provider["compat"]["sessionAffinityFormat"] == "openrouter"
    assert model["contextWindow"] == 4096
    assert model["maxTokens"] == 512
    assert model["samplingParams"] == {"temperature": 0, "top_p": 1}


def test_v1_pi_tools_have_no_shell_and_bound_paths_reads_and_results():
    text = TOOLS.read_text()
    for name in ("calculator", "search", "read", "list"):
        assert f'name: "{name}"' in text
    for required in (
        "MAX_READ_BYTES = 8192",
        "MAX_RESULTS = 256",
        "confinedExistingPath",
        "realpath",
        "path escapes the task root",
        "resolved path escapes the task root",
    ):
        assert required in text
    for forbidden in (
        "child_process",
        "exec(",
        "spawn(",
        "pi.exec",
        "eval(",
    ):
        assert forbidden not in text
