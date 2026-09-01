from scripts.build_e97_real_repo_holdout import SCHEMA, SPECS


def test_real_repository_holdout_pins_whole_repositories_and_minimal_repairs():
    assert SCHEMA == "emender-e97-real-repo-holdout-v1"
    assert len(SPECS) == 4
    repositories = set()
    for identity, spec in SPECS.items():
        assert identity
        assert spec["repository"] not in repositories
        repositories.add(spec["repository"])
        assert spec["url"].startswith("https://github.com/")
        assert len(spec["commit"]) == 40
        int(spec["commit"], 16)
        assert spec["path"]
        assert spec["clean"] != spec["mutated"]
        assert spec["focused_test"]
        assert "minimal" in spec["prompt"]
