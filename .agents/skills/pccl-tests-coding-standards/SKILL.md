---
name: pccl-tests-coding-standards
description: Applies the stable coding, compatibility, benchmark-integrity, licensing, and validation requirements of pccl-tests. Use before writing, modifying, or reviewing source code, build files, scripts, or documentation in this repository.
---

# pccl-tests Coding Standards

## Scope

These standards apply to every change in this repository. Requirements stated with *must* are mandatory; guidance stated with *should* is recommended.

Follow the style and abstractions of the surrounding pccl-tests code. pccl-tests is derived from NVIDIA's nccl-tests, so preserve the upstream-compatible structure and behavior unless the change intentionally introduces and documents a divergence. Do not reformat or refactor unrelated code.

## Repository and Build Integration

- Keep implementation, declarations, build definitions, tests, and documentation consistent with one another.
- Each collective benchmark lives in its own `src/<op>.cu` and shares the harness in `src/common.cu` and `src/common.h`. A new benchmark is registered in the `src` Makefile target list and exposes the weak-symbol `ncclTestEngine` override the same way the existing benchmarks do.
- Add new source files and dependencies to the established Make build driven by `build.sh`. Do not introduce a dependency when the existing standard library or project infrastructure is sufficient.
- Preserve the supported build modes: `rel_sym` and `dbg`; the single-process and `--use_mpi` multi-process builds; and the `PPU_ARCHS` targets. Platform- or version-specific behavior must be capability- or version-gated and must not break other supported configurations.
- Guard PCCL-version-specific code with `#if PCCL_VERSION_CODE >= PCCL_VERSION(...)` and provide the fallback for older versions, matching the existing guards.
- Build and shell changes must preserve the currently supported tool and shell environments, quote external input, and avoid unsafe command evaluation.

## Licensing and Provenance

- pccl-tests is licensed under BSD-3-Clause. `LICENSE.txt` must remain the unmodified BSD-3-Clause text with the NVIDIA copyright and the T-Head modifications notice.
- New original source, build, and script files must use the established SPDX header for their provenance: files derived from nccl-tests retain the NVIDIA copyright and the T-Head modifications notice, and new T-Head-authored files carry the T-Head copyright. Both use `SPDX-License-Identifier: BSD-3-Clause` and reference `LICENSE.txt`.
- Public files must not contain credentials, tokens, internal links, private addresses, private paths, or inaccessible dependencies.

## Style and Documentation

- Match the formatting, naming, file organization, and language conventions of the surrounding code and of the nccl-tests upstream where the code is shared.
- Use clear names that distinguish bytes and element counts, ranks and devices, in-place and out-of-place buffers, and send and receive counts where those distinctions affect correctness.
- Comments explain intent, invariants, ownership, synchronization, and compatibility constraints rather than restating the code.
- Source comments and public documentation use professional English and are updated with the behavior they describe.
- Do not combine functional changes with unrelated formatting or cleanup.

## Correctness and Error Handling

- Use the established `HGGCCHECK`, `PCCLCHECK`, and `TESTCHECK` macros for runtime, PCCL, and harness calls. Check every relevant return value and preserve the hostname, file, and line context they report.
- Release resources on normal and partial-failure paths. Communicators, buffers, streams, events, registrations, and MPI-owned resources must outlive the operations that use them.
- Validate user input, sizes, counts, roots, ranks, offsets, and optional interfaces before use. Reject invalid input before it can affect communicator initialization or measurement.
- Unsupported hardware, datatypes, ops, or PCCL-version capabilities must be reported or skipped clearly rather than failing after a benchmark starts.
- Correctness checking stays enabled by default (`-c 1`) and runs against the `verifiable` reference; a successful run reports `#wrong` equal to zero.

## Benchmark Integrity

- A result must describe the operation that was actually executed and the bytes that were actually transferred.
- `algbw` is the message size divided by the operation time. `busbw` applies the collective-specific correction factor documented in `doc/PERFORMANCE.md` (for example `2*(n-1)/n` for AllReduce and `(n-1)/n` for AllGather, ReduceScatter, and AlltoAll). Timing boundaries, synchronization, transfer direction, and these formulas must remain internally consistent.
- Do not add allocation, initialization, verification, diagnostics, output formatting, or unrelated synchronization to a measured interval unless the metric explicitly includes it. The selected timing mode (`--elapsed_type`: CPU, HGGC event, or HGPTI) must match what the reported number claims to measure.
- Keep automatically derived or benchmark-specific state local so that one benchmark or message size cannot change the behavior of a later one.
- Changes to a measurement path require before-and-after evidence appropriate to the affected hardware, timing mode, and execution modes.

## Interfaces and Output

- Treat CLI options and short flags, environment variables such as `NCCL_TESTS_SPLIT_MASK`, output column labels, report modes (`--report_type`), and any structured output as compatibility-sensitive interfaces.
- An intentional interface change must include validation, documentation in `README.md` or `doc/PERFORMANCE.md`, and a compatibility statement.
- The default report and any structured output must describe the same benchmark, ranks, sizes, time, `algbw`, `busbw`, and `#wrong`.
- New benchmarks and options must be discoverable through `--help` and documented in `README.md`.
- Multi-process (MPI) paths must preserve a consistent collective order and failure path across ranks so that no rank waits indefinitely.

## Validation

Validation must be proportional to the change:

| Change | Required validation |
|---|---|
| Source or build logic | Build every affected supported configuration (`rel_sym`, and `dbg`/`--use_mpi` when relevant) with no new warnings |
| Collective or data path | Run the affected benchmark with correctness checking and report `#wrong`, plus relevant size, datatype, and op boundaries |
| CLI or output contract | Check valid and invalid input, `--help`, and consistency between the default report and any structured output |
| Performance-sensitive path | Compare before-and-after `algbw`/`busbw` for the isolated benchmark and the relevant full sequence, with the timing mode stated |
| Multi-process behavior | Build with `--use_mpi` and run the affected multi-rank path when the environment is available |
| Documentation or licensing | Check commands, options, links, headers, notices, formatting, and public-repository safety |

Never report a build, test, platform, or performance result that was not obtained. State unavailable hardware, MPI, or multi-node validation explicitly in the pull request description.
