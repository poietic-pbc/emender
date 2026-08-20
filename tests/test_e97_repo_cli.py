import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/e97_repo_cli.py"


def run(cwd: Path, *args: str, ok: bool = True):
    result = subprocess.run([sys.executable, str(CLI), *args], cwd=cwd, text=True, capture_output=True)
    assert (result.returncode == 0) is ok, result.stderr
    return json.loads(result.stdout) if ok else result


def test_repo_cli_list_count_search_read_and_json(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("first\ntokenizer = 'p50k_base'\n")
    (tmp_path / "src/b.py").write_text("other\n")
    (tmp_path / "config.json").write_text('{"model":{"depth":11,"tokenizer":"p50k_base"}}')

    listing = run(tmp_path, "list", "--path", "src", "--pattern", "*.py")
    assert [row["path"] for row in listing["entries"]] == ["src/a.py", "src/b.py"]
    assert run(tmp_path, "count", "--path", "src", "--pattern", "*.py")["count"] == 2
    assert run(tmp_path, "search", "--path", "src", "--pattern", "*.py", "--query", "tokenizer")["matches"] == [
        {"line": 2, "path": "src/a.py", "text": "tokenizer = 'p50k_base'"}
    ]
    assert run(tmp_path, "read", "--path", "src/a.py", "--start", "2", "--end", "2")["lines"] == [
        {"line": 2, "text": "tokenizer = 'p50k_base'"}
    ]
    assert run(tmp_path, "json", "--path", "config.json", "--pointer", "/model/depth")["value"] == 11


def test_repo_cli_rejects_escape_and_bounds_results(tmp_path: Path):
    (tmp_path / "inside").write_text("ok")
    escaped = run(tmp_path, "read", "--path", "../outside", ok=False)
    assert "escapes" in escaped.stderr or "No such file" in escaped.stderr
    for number in range(3):
        (tmp_path / f"{number}.txt").write_text("x")
    bounded = run(tmp_path, "list", "--max-results", "2", ok=False)
    assert "result limit exceeded" in bounded.stderr


def test_repo_cli_help_is_discoverable(tmp_path: Path):
    result = subprocess.run([sys.executable, str(CLI), "--help"], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0
    for command in ("list", "count", "read", "search", "json"):
        assert command in result.stdout
