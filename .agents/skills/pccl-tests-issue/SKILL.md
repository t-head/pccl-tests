---
name: pccl-tests-issue
description: Create and triage pccl-tests GitHub issues, the only channel for task management and bug tracking, which every pull request must reference. Use before opening any pull request that has no open issue to reference, including chores, documentation, tests, and refactors, and when writing acceptance criteria or a bug report.
---

# pccl-tests Issue Guide

## Scope

GitHub issues are the only channel for pccl-tests task management and bug tracking. Every pull request references at least one open issue with `Closes #<issue>` or `Refs #<issue>`. When no open issue covers the change, create one before opening the pull request.

Issues and their attachments are public. Do not include credentials, tokens, internal links, private absolute paths, or internal-only identifiers. Redact sensitive hostnames and addresses from logs before attaching them, but keep the technical content intact.

Search existing issues first and link related discussions instead of opening a duplicate.

## Issue types

Classify every issue as one of three types and select that type when creating it:

| Type | Use |
|---|---|
| `Feature` | A new benchmark, collective, option, correctness check, or improvement |
| `Bug` | Unexpected behavior, wrong results, a crash, a hang, or a build failure |
| `Task` | Planned maintenance or a small change such as a version bump |

## Common content

Every issue states, clearly and specifically:

1. **Requirement or problem**: what is needed, or what went wrong.
2. **Background**: the use case, the affected users or scenarios, and the expected benefit or impact.

## Feature

A `Feature` issue also states:

3. **Test acceptance criteria**: the measurable conditions that define completion, including:
   - the Zhenwu PPU platforms the change targets, for example Zhenwu M890;
   - how to build and run the check, with the `build.sh` options, the benchmark executable, its arguments, the MPI launch when relevant, and any environment variable;
   - the expected result: for correctness, the expected `#wrong` count; for performance, the metric (`algbw` or `busbw`), the message sizes, ranks, timing mode, and the baseline used for comparison.

The acceptance criteria must describe the same tests that the `Validation` section of the pull request closing the issue reports. Do not state one test in the issue and a different test in the pull request.

## Bug

A `Bug` issue also states:

3. **Version**: the pccl-tests release tag or commit, and the PCCL library version the benchmarks were built against.
4. **Environment**: the T-Head SAIL SDK version, the OS, the PPU model and arch (`ppu_10` or `ppu_15`), the node count, the PPUs per node, and the Open MPI version for a multi-process build.
5. **Reproduction**: the `build.sh` options, the benchmark executable and its arguments, the MPI launch and rank count, the relevant environment variables, and how often it fails.
6. **Expected and actual behavior**: what should happen, what happened, and the first error or warning. For a correctness failure, report the `#wrong` count and the affected collective, datatype, op, and message size.
7. **Original log file**: the console log or diagnostic output as the run produced it, not a curated excerpt, with the sensitive hostnames and addresses redacted as described in Scope. For a correctness failure, attach the per-rank file written by `--report_check_filename <prefix>` (`<prefix>_<rank>.log`).

## Task

A `Task` issue also states:

3. **Acceptance criteria**: the measurable conditions for completion, or `None` for a small change such as a version bump.

## Create the issue

Create the issue on GitHub under Issues and choose the Feature, Bug, or Task form under `.github/ISSUE_TEMPLATE/`; the form sets the type and prompts for the fields above. The non-interactive command below creates the body directly and does not set the type, so set the type on the created issue afterward, or run `gh issue create` with no arguments to pick a template interactively:

```bash
gh issue create --title "<summary>" --body-file issue.md --assignee "@me"
```

Every issue is assigned at creation to the author, the person on whose behalf the agent acts; assigning a different person is an explicit, manual adjustment. When the creating account cannot assign, leave the issue unassigned and report that a maintainer needs to assign it.

Record the issue number and reference it in the pull request description as `Closes #<issue>` or `Refs #<issue>`.

## Before opening the pull request

- The issue states the requirement or problem and the background.
- A `Feature` or `Task` issue states the acceptance criteria; a small `Task` may use `None`.
- A `Bug` issue states the version and attaches the original log.
- The acceptance criteria match the tests that the pull request `Validation` section reports.
- The pull request references the issue with `Closes #<issue>` or `Refs #<issue>`.
