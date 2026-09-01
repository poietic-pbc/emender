# E97 4B real-repository holdout v1

**Status:** frozen and mechanically validated before public post-training payload download

The holdout contains four injected regressions on immutable public repository
commits. Whole repository identities are excluded from every later SFT,
teacher-generation, rejection-sampling, and RL source.

## Authority

- root: `/mnt/nvme1n1/erikg/evals/e97-real-repo-holdout-v1`;
- tasks: 4;
- manifest SHA-256:
  `939bcd66768884a1e5ec44bcf11fdf5d09602ebdabe86d4caefffd4ace8dbb60`;
- task payload SHA-256:
  `3b93fc433cc6e23e98bfcd62b294f5d5c2e5d4144bfe3983cd661f53b2414676`.

| Repository | Commit | Git archive SHA-256 | Task |
|---|---|---|---|
| `pallets/markupsafe` | `b2e4d9c7687be25695fffbe93a37622302b24fb1` | `2e8b0f71c53813a83195e50dc0445f13e5ddcfba720f118870c594a43f12a435` | restore entity decoding in `Markup.striptags` |
| `python-humanize/humanize` | `3201e702ed7eae506f793fad0aec204f387aeb4c` | `fa15e42fc3ccb85931c35febf7a9b9d08de3abf75d90fb2d93d4e9c536edd67a` | restore decimal/binary filesize base selection |
| `more-itertools/more-itertools` | `2fe1b2eeb9d75f994113fe3ac76d14b6bcd6fb10` | `7f647319a2b47cc4d2743c6d34761fa3b6823bb1d533651e47874a86366eee4d` | restore requested `chunked` size |
| `prettytable/prettytable` | `2a6cd4fb41bc6754eac57b43fc6dbd43b08ae368` | `be9f7908e40707893d8bd7233a810d49926a94b5dfd1cb5cf3c68e69975e7baa` | restore terminal ANSI reset in `ColorTable` output |

## Validation

`scripts/validate_e97_real_repo_holdout.py` copied every pinned source, applied
only the declared mutation, and verified:

1. the focused test or command failed after mutation;
2. only the intended tracked source file changed;
3. the exact minimal repair restored the source;
4. the focused test or command then exited zero;
5. no tracked diff remained.

All four mutations failed as intended and all four expected repairs passed.
Failure and repair output digests are retained in
`e97-4b-real-repository-holdout-v1-validation.json`.

The evaluation harness must score functional focused-test success, bounded diff,
sandbox safety, and grounded completion. An exact canonical tool sequence is
reported for diagnosis but is not required when an alternative sequence yields
the same minimal verified repair.
