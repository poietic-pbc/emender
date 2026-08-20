from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_e97_sandbox_cli.py"
TOOLS = ROOT / "configs/pi/e97-cli-tools.ts"
DEFINITION = ROOT / "configs/pi/e97-cli-sandbox.def"


def test_runner_passes_argv_without_host_shell_and_uses_qualified_boundary():
    text = RUNNER.read_text()
    for required in (
        '"apptainer", "--silent", "exec"',
        '"--containall"',
        '"--cleanenv"',
        '"--net", "--network", "none"',
        '"--no-privs", "--drop-caps", "all"',
        '"--no-mount", "bind-paths,home,cwd,tmp,hostfs,proc,sys"',
        'f"{cwd}:/work:rw"',
        '"--cwd", "/work"',
        '"/usr/bin/prlimit"',
        '"--core=0"',
        '"--cpu=60"',
        '"--fsize=1048576"',
        '"--nofile=256"',
        "start_new_session=True",
        "os.killpg",
        "subprocess.DEVNULL",
        "sha256(image)",
    ):
        assert required in text
    assert "shell=True" not in text


def test_pi_cli_tool_has_one_argv_interface_and_grounded_termination():
    text = TOOLS.read_text()
    for required in (
        'name: "cli"',
        'name: "submit_answer"',
        "EMENDER_CLI_IMAGE",
        "EMENDER_CLI_IMAGE_SHA256",
        "EMENDER_PYTHON",
        "execFileAsync",
        "ctx.cwd",
        "maxItems: 64",
        "row.stdout.includes(value)",
        "row.stdout.includes(evidence)",
        "terminate: true",
    ):
        assert required in text
    assert "shell:" not in text


def test_image_definition_contains_reviewed_discovery_tools():
    text = DEFINITION.read_text()
    for required in (
        "From: python:3.12-slim-bookworm",
        "scripts/e97_repo_cli.py /usr/local/bin/repo",
        "git",
        "jq",
        "ripgrep",
        "util-linux",
        "chmod 0755 /usr/local/bin/repo",
    ):
        assert required in text
