# FLIP upgrade and re-enable gate audit (2026-07-17)

Task: `upgrade-reenable-flip-route-fix`

## Outcome

The upgrade and FLIP re-enable were intentionally not performed because the
required upstream review/merge gate is not satisfied. The fixed branch exists,
but its patch is not contained in upstream `main`, and no GitHub pull request
exists for the branch.

No emender configuration, installed WG binary, or Slurm job was changed.

## Upstream identity and merge check

The predecessor's validation artifact identifies:

- branch: `fix/pending-eval-flip-qualified-route`
- patch commit: `bef69590` (`fix(agency): preserve FLIP execution routes`)
- branch tip: `7d44b52f0a010e7300843baeb6cb02afb431a324`

On 2026-07-17, after fetching both refs from `graphwork/wg`, the remote refs
resolved to:

```text
origin/main                                      528b8ee0258bc48c6fd11ad9b87562610a276526
origin/fix/pending-eval-flip-qualified-route     7d44b52f0a010e7300843baeb6cb02afb431a324
```

The containment query returned only the feature branch:

```bash
git branch -r --contains bef69590
```

```text
origin/fix/pending-eval-flip-qualified-route
```

The GitHub API query returned an empty array:

```bash
curl -fsSL \
  'https://api.github.com/repos/graphwork/wg/pulls?head=graphwork:fix/pending-eval-flip-qualified-route'
```

```json
[]
```

Thus there is neither evidence of review nor a merge commit containing the
fix. The predecessor artifact's URL is a pull-request creation URL, not an
opened or merged pull request.

## Installed build and containment

Before the audit, the installed executable was:

```text
/ccs/home/erikgarrison/.cargo/bin/wg
wg 0.1.0
```

It was not replaced because the task requires installing the exact reviewed
and merged revision. The live emender FLIP setting was not changed, and no
disposable lifecycle tests were started with an unapproved build. No Slurm
inspection or mutation commands were issued.

## Resume procedure

Once an upstream pull request is reviewed and merged:

1. Fetch `graphwork/wg` and record the merged `origin/main` revision.
2. Verify `git merge-base --is-ancestor bef69590 origin/main` exits zero.
3. Build and install that exact recorded revision.
4. Enable FLIP in a newly created disposable graph only.
5. Run three ordinary parent task lifecycles and observe each automatic
   `FLIP -> evaluator -> Done` chain without manually completing hidden tasks.
6. Restart the disposable dispatcher and retry one lifecycle; verify there is
   exactly one evaluator task and one evaluation result for the parent.
7. Only after all disposable checks pass, enable FLIP in emender and append the
   exact installed commit and configuration diff to this audit.

