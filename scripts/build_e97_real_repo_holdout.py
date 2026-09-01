#!/usr/bin/env python3
"""Freeze pinned real-repository injected-regression evaluation tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from ndm.data.masked_sft_dataset import sha256

SCHEMA = "emender-e97-real-repo-holdout-v1"
SPECS = {
    "markupsafe-striptags-unescape": {
        "repository": "markupsafe",
        "url": "https://github.com/pallets/markupsafe.git",
        "commit": "b2e4d9c7687be25695fffbe93a37622302b24fb1",
        "path": "src/markupsafe/__init__.py",
        "clean": "        return self.__class__(value).unescape()",
        "mutated": "        return value",
        "focused_test": "PYTHONPATH=src python -m pytest -q tests/test_markupsafe.py",
        "setup_files": {},
        "prompt": (
            "Markup.striptags no longer decodes escaped HTML entities. Run the focused "
            "MarkupSafe test, diagnose the implementation regression, make the minimal "
            "source correction, and rerun the focused test."
        ),
    },
    "humanize-filesize-base-selection": {
        "repository": "humanize",
        "url": "https://github.com/python-humanize/humanize.git",
        "commit": "3201e702ed7eae506f793fad0aec204f387aeb4c",
        "path": "src/humanize/filesize.py",
        "clean": "    base = 1024 if (gnu or binary) else 1000",
        "mutated": "    base = 1000 if (gnu or binary) else 1024",
        "focused_test": "PYTHONPATH=src python -m pytest -q tests/test_filesize.py",
        "setup_files": {"src/humanize/_version.py": "__version__ = \"holdout\"\n"},
        "prompt": (
            "Humanize's decimal and binary filesize modes are selecting the wrong bases. "
            "Run the focused filesize tests, locate the regression, restore the intended "
            "mode selection with a minimal edit, and rerun the tests."
        ),
    },
    "more-itertools-chunk-size": {
        "repository": "more-itertools",
        "url": "https://github.com/more-itertools/more-itertools.git",
        "commit": "2fe1b2eeb9d75f994113fe3ac76d14b6bcd6fb10",
        "path": "more_itertools/more.py",
        "clean": "    iterator = iter(partial(take, n, iter(iterable)), [])",
        "mutated": "    iterator = iter(partial(take, n - 1 if n is not None else n, iter(iterable)), [])",
        "focused_test": "PYTHONPATH=. python -m pytest -q tests/test_more.py::ChunkedTests",
        "setup_files": {},
        "prompt": (
            "more_itertools.chunked is producing chunks smaller than the requested size. "
            "Run the focused ChunkedTests, inspect the implementation, correct the size "
            "regression minimally, and rerun the focused tests."
        ),
    },
    "prettytable-color-reset": {
        "repository": "prettytable",
        "url": "https://github.com/prettytable/prettytable.git",
        "commit": "2a6cd4fb41bc6754eac57b43fc6dbd43b08ae368",
        "path": "src/prettytable/colortable.py",
        "clean": "        return super().get_string(**kwargs) + RESET_CODE",
        "mutated": "        return super().get_string(**kwargs)",
        "focused_test": (
            "PYTHONPATH=src python -c 'from prettytable.colortable import ColorTable, RESET_CODE; "
            "from prettytable.prettytable import PrettyTable; "
            "PrettyTable.get_string=lambda self,**kwargs:\"BODY\"; "
            "x=ColorTable.__new__(ColorTable); assert x.get_string() == \"BODY\" + RESET_CODE'"
        ),
        "setup_files": {"src/prettytable/_version.py": "__version__ = \"holdout\"\n"},
        "prompt": (
            "ColorTable.get_string no longer terminates rendered output with the ANSI "
            "reset sequence. Run the focused command, diagnose the regression, make the "
            "minimal source fix, and rerun the command."
        ),
    },
}


def archive_sha256(repository: Path, commit: str) -> str:
    archive = subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", commit],
        check=True, stdout=subprocess.PIPE).stdout
    return hashlib.sha256(archive).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    tasks_path = args.output_root / "tasks.jsonl"
    repositories = {}
    with tasks_path.open("w") as stream:
        for identity, spec in SPECS.items():
            repository = args.source_root / spec["repository"]
            commit = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
            if commit != spec["commit"]:
                raise RuntimeError(f"{identity}: commit mismatch")
            status = subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                check=True, text=True, stdout=subprocess.PIPE).stdout
            if status:
                raise RuntimeError(f"{identity}: source repository is dirty")
            text = (repository / spec["path"]).read_text()
            if text.count(spec["clean"]) != 1:
                raise RuntimeError(f"{identity}: mutation precondition mismatch")
            archive = archive_sha256(repository, commit)
            repositories[spec["repository"]] = {
                "url": spec["url"], "commit": commit, "git_archive_sha256": archive,
                "training_exclusion": "exclude entire repository identity from all post-training sources",
            }
            row = {"id": identity, "split": 1, **spec,
                   "expected_patch": {"path": spec["path"],
                                      "oldText": spec["mutated"],
                                      "newText": spec["clean"]}}
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": SCHEMA, "status": "complete",
        "purpose": "real-repository injected-regression post-training holdout",
        "tasks": len(SPECS), "repositories": repositories,
        "training_exclusion": (
            "frozen before public post-training payload download; exclude whole repository "
            "identities, all tasks, mutations, expected patches, prompts, and test outputs"),
        "outputs": {"tasks": {"path": str(tasks_path.resolve()),
                                "bytes": tasks_path.stat().st_size,
                                "sha256": sha256(tasks_path)}},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
