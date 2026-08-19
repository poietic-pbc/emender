import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs/pi/e97-dense-agent.models.json"
TOOLS = ROOT / "configs/pi/e97-v1-tools.ts"
V2_TOOLS = ROOT / "configs/pi/e97-v2-tools.ts"
V3_TOOLS = ROOT / "configs/pi/e97-v3-tools.ts"


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
    # Pi reserves 4096 context tokens before clamping max output. E97 needs a
    # larger advertised window so bounded 4K transcripts retain output space.
    assert model["contextWindow"] == 8192
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


def test_v2_tools_return_typed_values_and_require_grounded_termination():
    text = V2_TOOLS.read_text()
    for name in ("calculator", "lookup", "count", "submit_answer"):
        assert f'name: "{name}"' in text
    for required in (
        "expectedAnswer",
        "value !== expectedAnswer",
        "latest tool result",
        "terminate: true",
        "MAX_READ_BYTES = 8192",
        "realpath",
    ):
        assert required in text
    for forbidden in ("child_process", "exec(", "spawn(", "pi.exec", "eval("):
        assert forbidden not in text


def test_v3_lookup_tools_encode_the_requested_field_in_the_tool_name():
    text = V3_TOOLS.read_text()
    assert 'name: `lookup_${field}`' in text
    assert '["owner", "budget"] as const' in text
    assert 'parameters: Type.Object({ project:' in text
    assert 'name: "submit_answer"' in text
    for forbidden in ("child_process", "exec(", "spawn(", "pi.exec", "eval("):
        assert forbidden not in text
