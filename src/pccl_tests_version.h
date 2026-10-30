/*************************************************************************
 * Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Test-suite versioning, mirroring the nccl-tests NCCL_TESTS_VERSION scheme:
 * a manually bumped release version, plus a git revision injected at build
 * time by src/Makefile (-DPCCL_TESTS_GIT_VERSION=...).
 */
#ifndef PCCL_TESTS_VERSION_H__
#define PCCL_TESTS_VERSION_H__

#ifndef PCCL_TESTS_VERSION
#define PCCL_TESTS_VERSION "2.3.0"
#endif

#ifndef PCCL_TESTS_GIT_VERSION
#define PCCL_TESTS_GIT_VERSION "unknown"
#endif

#endif /* PCCL_TESTS_VERSION_H__ */
