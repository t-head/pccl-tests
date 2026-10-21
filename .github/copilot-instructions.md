# pccl-tests Code Review Instructions

Review every pull request in this repository according to the following rules.

## Review scope

- For the first review, review the pull request diff and the code paths that it affects. For a subsequent push, review only the incremental diff from the last reviewed head SHA to the current head SHA, including metadata changes relevant to that diff.
- Before a subsequent review, identify both head SHAs and read the existing review threads and their classifications. After amend, force-push, or rebase, compare the two final trees rather than treating the entire base-to-head patch as new work.
- Do not repeat a finding already classified as `addressed`, `won't fix`, or `incorrect` unless the incremental change invalidates its recorded resolution. Report a new finding on unchanged code only when the incremental change makes it reachable, worsens it, or depends on it.
- If the previous reviewed revision or review history is unavailable, state that incremental review could not be verified and request that context; do not silently claim an incremental review or restart a full review.
- Read each incrementally changed file in full for context and trace the referenced definitions before a finding is reported; reading context does not expand the finding scope.
- Unrelated existing code is not reported unless the change makes it reachable, worsens it, or depends on it.
- The `main` branch and maintained release branches are protected branches. The review covers the code and metadata proposed in the pull request.

## Review procedure

- Identify the review baseline and current head SHA, and read the actual diff for the scope above. Recheck existing unresolved findings affected by that diff.
- Read the changed files in full and inspect the affected declarations, callers, tests, build files, and documentation.
- Evaluate collective, kernel-launch, synchronization, and memory-ordering behavior against the PCCL, HGGC, and PPU implementation rather than assumed NCCL or CUDA behavior.
- Verify that the commit message and the pull request description accurately describe the diff, including the stated tests and compatibility impact.
- Classify findings as Blocker, Major, or Normal, in that order.
- Each finding includes the severity and a concise title, the exact `file:line` evidence, the concrete failure mode and the affected configuration, the reason the change causes the failure, a bounded correction, and the required validation.
- A runtime failure is reported only when a reachable code path exists. An uncertain issue is classified as Normal, and the author is asked for evidence.
- When no findings exist, the review states so explicitly and lists validation gaps separately.

## Severity model

- Blocker (merge is not permitted) applies only when the change can cause wrong benchmark or correctness results, silent data corruption, memory safety violations, hangs, security or license violations, build failure in a supported configuration, or public disclosure of internal information, or when the commit message misdescribes the change.
- Major (correction is required) applies when the change causes a material regression, a compatibility break, a resource leak, a misleading diagnostic, incomplete integration, or insufficient validation of a high-risk path, or when the pull request description misdescribes the change. A Major finding remains unresolved at merge only with a documented maintainer decision.
- Normal (the author provides an explanation) applies to maintainability, clarity, weak evidence, minor inconsistency, or behavior that may be intentional but requires an explicit rationale. Style preferences are not classified as Major.
- When findings are posted as pull request comments, Blocker and Major findings are prefixed with `Required:`, Normal findings with `Question:`, and optional improvements with `Suggestion:`.

## Accepted project conventions (not reported as findings)

- The `.cu` and `.cuh` extensions belong to the PPU hgcc build and do not indicate a CUDA runtime dependency. HGGC, HGPTI, and `hggc*` names refer to the PPU runtime and profiling interfaces.
- The NVIDIA copyright notices, the BSD-3-Clause license, and the statement that pccl-tests is derived from nccl-tests are required attribution.
- `NCCL_TESTS_*` environment variable names and the `ncclTestEngine`/`NCCL_WEAK` symbols are retained from the nccl-tests upstream for source and output compatibility; they do not indicate an NCCL runtime dependency.
- `#if PCCL_VERSION_CODE >= PCCL_VERSION(...)` guards gate version-specific collective and device-API paths. A guarded path with a fallback for older versions is not a defect.
- Device-side implementations (`-D`/`--device_implementation`, symmetric registration `-R 2`) require PCCL 2.2 or later; support varies by benchmark and is gated rather than a defect.
- Collective bandwidth arrows, column labels, and the `busbw` correction factors follow the nccl-tests conventions and `doc/PERFORMANCE.md`, retained for output compatibility. Only regressions, internally inconsistent new labels, or undocumented contract changes are reported.
- `AGENTS.md` and `.agents/skills/<skill-name>/SKILL.md` use GitHub's standard agent instruction and Agent Skills paths. `copilot-instructions.md` supplies review-wide guidance; the skills supply task-specific guidance.
- The prechecks run the checkers from the pull request working tree. Whether a pull request may modify a workflow or a checker is a maintainer review decision, not a process-enforced gate; do not report the absence of tamper-proofing as a finding. Review a workflow or checker change for correctness like any other diff.
- The DCO sign-off requires a trailing `Signed-off-by: Name <email>` trailer whose name contains no angle brackets and whose email has exactly one `@`. The email is intentionally permissive so that anonymous and no-reply addresses are accepted; it is not matched against the commit author and not validated against strict RFC dot-atom rules. Report only a structurally malformed trailer, not a permissive but valid address.

## Lessons from prior reviews

These cases define the intended checker behavior, not exemptions for regressions. Before reporting a similar issue, verify the current implementation and demonstrate the failing input; group related boundary cases in one finding rather than rediscovering them over successive reviews.

- A key-feature list starts with `- ` (hyphen-space); a lone `-` is an empty item. Prose beginning with `--option` or `-1` is not a list and is not checked by requirement 7.
- Issue-reference candidates occupy the complete line. Prose such as `Fix issue 12 by ...` is not a footer reference; standalone `Closes issue 12` and `Closes #abc` are malformed references. A reference belongs in the final contiguous footer, not before later prose. The footer must contain a reference; a sign-off-only footer is rejected as a missing reference.
- Issue numbers are accepted only as ASCII digits; a reference candidate containing a non-ASCII digit such as an Arabic-Indic numeral is still detected and reported as malformed rather than silently treated as prose or converted to the corresponding ASCII issue.
- A typed commit title's colon may be followed by one or more spaces; extra whitespace does not bypass the subject, scope, or non-descriptive-title checks.
- A sign-off alone is not a commit title. DCO identities must come from the contiguous trailing sign-off block; an author's sign-off elsewhere in the body does not satisfy DCO.
- `WIP` is a standalone word or prefix form (`wip:`, `wip-`, `wip_`), not a substring of valid subjects such as `wipe stale cache`.
- PR descriptions contain each required heading exactly once. Change types use the supported labels and the exact commit-type mapping, while an unprefixed title does not impose a type mapping.
- Markdown checks operate on parsed structure. Fenced and indented code blocks, multiline code spans, and comments must not supply headings or selected checkboxes; literal comment markers inside code must not hide later sections. Cover these cases together in the checker regression suite.
- Online issue validation uses only plain references in `Related issue`, queries the PR's base repository, and excludes pull request objects. An open issue mentioned elsewhere does not satisfy the requirement; API failures must not be reported as a successful lookup.
- Local fork validation prefers the fetched `upstream` target over `origin`; Actions can use `origin`. Explicit `--base` takes precedence.
- Detailed policy lives in the task-specific skills; CONTRIBUTING.md summarizes the workflow. Do not report consistent summaries as conflicting policy or request duplicate rule files.

## Blocker: Benchmark and correctness integrity

- Reject any path that can report bandwidth or latency for bytes that were not transferred, or that can report `#wrong` as zero when the result was not verified.
- Validate every size, count, offset, root, rank, stride, and loop calculation for overflow, truncation, division by zero, omitted tails, in-place/out-of-place overlap, and out-of-bounds access.
- Message-size, iteration, warm-up, root, and device-count options reject invalid input before communicator initialization and measurement.
- The `busbw` correction factor matches the collective and the rank count as documented in `doc/PERFORMANCE.md`; `algbw` is size divided by the measured operation time. A factor is never applied to the wrong collective or averaged across ranks by inference.
- A reduction or rooted collective uses the actual op, datatype, and root; the correctness reference in `verifiable` matches the operation that was executed.
- Default correctness checking (`-c`) is not silently bypassed, narrowed, or moved before the completion of the measured collective operations.

## Blocker: Memory, resources, and asynchronous lifetime

- Reject use-after-free, double free, double release, mismatched allocation and deallocation interfaces, and resource leaks that can exhaust resources or corrupt a supported benchmark run.
- Communicators, buffers, streams, events, registrations, graphs, and MPI-owned resources are destroyed or freed on normal and partial-failure paths.
- Host objects and device buffers outlive queued collective and device-side work and all synchronization points that consume them.
- Stack objects, temporary vectors, and references do not escape into asynchronous execution.
- Unsupported datatypes, ops, device implementations, or hardware capabilities are reported or skipped through the established mechanism rather than executed optimistically.

## Blocker: PCCL version and ABI safety

- Version-specific PCCL fields, device-API structures, and EIT members are used only inside the matching `PCCL_VERSION_CODE >= PCCL_VERSION(...)` guard, with a fallback for older versions.
- Reject reads of optional device-API fields or function pointers that are not present in the compiled or runtime PCCL version.
- Public structures, enum values, exported symbols, option semantics, and environment variable meanings do not change incompatibly without an explicit compatibility plan.

## Blocker: MPI and hang safety

- All ranks enter collective operations, barriers, waits, and shutdown paths in a consistent order.
- Fatal multi-process errors converge through the existing abort or timeout mechanism so that no rank waits indefinitely.
- Reject rank-dependent branches that change the number or order of collectives without a proven protocol.
- Rank, local device, hostname, and group attribution (including `NCCL_TESTS_SPLIT_MASK` grouping) remain correct in diagnostics and output.
- A timeout is not treated as a correctness mechanism.

## Blocker: Benchmark integration and output validity

- Every benchmark registers through the established weak-symbol `ncclTestEngine` mechanism and is selectable through the intended CLI path.
- A new benchmark or option is declared, implemented, registered, gated for unsupported configurations, and shown in `--help`.
- Unsupported capabilities produce a clear skip or report rather than a misleading runtime error.
- Text and any structured output describe the same benchmark, ranks, sizes, time, `algbw`, `busbw`, and `#wrong`. Reject malformed or unparsable structured output.

## Blocker: Build and CLI security

- Reject shell command injection, unsafe `eval`, unintended expansion of command substitutions, backticks, or variables, and argument parsing that corrupts quotes or whitespace in `build.sh` and the CI scripts.
- A supported build configuration does not lose source files, include paths, libraries, compile definitions, arch targets, or executable permissions across `rel_sym`, `dbg`, and `--use_mpi`.
- Reject hard-coded credentials, tokens, private keys, internal repositories, private absolute paths, and insecure download bypasses.

## Blocker: License and public-repository safety

- `LICENSE.txt` remains the unmodified BSD-3-Clause text with the NVIDIA copyright and the T-Head modifications notice.
- Derived source files retain the applicable NVIDIA copyright notice and the T-Head modifications notice; new original files use the established T-Head BSD-3-Clause SPDX header.
- Reject internal CI configuration, internal code review links, internal issue identifiers, internal download endpoints, secrets, build artifacts, temporary files, and inaccessible submodule or LFS objects.
- The README Copyright section retains the license statement and the NVIDIA and T-Head copyright notices.

## Blocker: CI gate integrity

- A new test added to `.github/workflows/scripts/casecmd.list` runs in CI as a test matrix entry and is covered by `PCCL Tests Gate` automatically; a new top-level workflow job is added to `tests-gate.needs`.
- A required job failure, cancellation, or an unexpected skip on a pull request fails `PCCL Tests Gate`; a green gate beside a needed job that did not succeed is a defect.
- The `test` matrix job expands only on `pull_request` events and is expected to be skipped on `push` and `workflow_dispatch`; the gate requires it to succeed on pull requests and tolerates its skip elsewhere. Do not report the non-PR skip as a gate defect.
- Changes to CI routing or gate rules are verified against both passing and failing cases: a clean run, a failed job, and the PR-only jobs on pull_request versus push events.

## Major: Performance validity and regression risk

- Report new synchronization, allocation, context switching, copies, logging, or driver queries in the measured hot path.
- Require before-and-after evidence for changes to collective algorithms, device-side kernels, registration or symmetric-memory paths, launch or CTA policy, timing mode, message-size sweep, iteration or aggregation structure, or bandwidth calculation.
- Compare a benchmark run alone with the full sequence when shared or global state may affect results.
- Distinguish a true performance change from the correction of a previously invalid measurement, and require both isolated and sequence results.
- Require single-process and multi-process (MPI) evidence when shared code affects both modes.
- Verification, buffer initialization, diagnostics, and output formatting remain outside the timed interval unless the metric explicitly includes them.

## Major: Result semantics and compatibility

- Changes to column labels, report modes, `busbw` factors, option names or short flags, `NCCL_TESTS_*` environment variables, or default values can break downstream parsing and require a compatibility statement.
- Per-collective, in-place/out-of-place, aggregate, and latency formulas match the bytes and directions that are actually measured.
- Diagnostics do not overstate root causes; a correctness failure reports the affected collective, datatype, op, size, and rank without implying an unproven cause.
- Attribution of a transfer is valid only when the source and the verified destination are independently readable and distinct buffers; otherwise only expected and actual values are reported.

## Major: Resource and error quality

- Report unbounded or per-iteration resource leaks even when they do not immediately corrupt results.
- New PCCL, HGGC, HGPTI, and MPI calls use the established `PCCLCHECK`, `HGGCCHECK`, and `TESTCHECK` wrappers or explicitly handle recoverable failures.
- Text and structured error paths preserve the original error and sufficient host, rank, and device context.

## Major: Build, documentation, and integration

- New source files are added to the `src` Makefile target list; new options have defaults, range checks, `--help` text, and documentation appropriate to their visibility.
- Build changes cover the normal, debug, stripped, MPI, and version-gated configurations that the diff affects.
- README and `doc/PERFORMANCE.md` commands, option names, defaults, collective names, model names, links, and bandwidth formulas match the current code.
- Performance examples state their measurement context (timing mode, ranks, sizes) and do not imply simultaneous aggregate throughput when the values were measured sequentially.

## Normal: Maintainability and consistency

- Request a rationale when ownership, context, direction, or synchronization invariants are not documented.
- Report stale or misleading comments, spelling errors, unclear byte and element naming, inconsistent terminology, and unexplained magic values.
- Request evidence for omitted tests; tests unrelated to the change are not required.
- Recommend splitting a pull request that mixes unrelated feature, refactoring, documentation, and formatting work or exceeds a reviewable size.
- Check formatting, include guards, executable bits, line endings, trailing whitespace, and consistency with the surrounding style and the nccl-tests upstream.
- Public code and documentation use professional English.

## Validation matrix

- Validation is proportional to the diff.
- CLI and argument parsing: valid and invalid boundary cases, `--help` output, and option/env interactions.
- C++ and device code: the affected build configurations (`rel_sym`, and `dbg`/`--use_mpi` when relevant) with no new warnings.
- Correctness: the affected collective with `-c 1` reporting `#wrong` equal to zero, plus intentional-mismatch and datatype/op boundaries where relevant.
- Performance: A/B comparison of `algbw`/`busbw` for the isolated benchmark and the full sequence, with the timing mode and rank count stated.
- MPI: at least two ranks or nodes, or a justified statement that the environment is unavailable; failure propagation is covered when it changes.
- Output: consistency of benchmark names, ranks, sizes, time, bandwidth, and `#wrong` between the default report and any structured output.
- Documentation and license: link, command, and option validation, SPDX and notice scan, and internal-information scan.
- Check-in scripts: run the offline checker regression suite, including Markdown structure, issue-state API error paths, commit format, and DCO; checking the current PR alone does not test the checkers.
- Runtime validation that was not executed is never claimed.

## Output format

- Findings are sorted by severity and impact in the sections Blocker, Major, and Normal, followed by the sections Validation gaps and Verdict. Empty severity sections are omitted.
- Each finding is written as `- [<severity>] <title> — path:line`, followed by the entries Impact, Evidence, Fix, and Validate.
- The Validation gaps section lists missing evidence and unavailable environments.
- The verdict is No-Go, Go after fixes, or Go. A Blocker always results in No-Go. A Major finding normally results in Go after fixes. A Normal finding requires an author response but does not block the merge by itself.
