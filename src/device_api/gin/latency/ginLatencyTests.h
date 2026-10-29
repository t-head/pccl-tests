/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include "common.h"

/*
 * Launch entry points for the latency tests, one per GIN operation, each
 * implemented by the like-named .cu file in this directory. The *_main.cu
 * runners select one of these according to --gin_op; the linker then pulls only
 * the referenced implementations out of the category archive.
 *
 * Every entry point matches ginRunFn_t so it can be assigned to
 * ginBenchmark_t::run directly.
 */

void pcclTestGinPutLatencyPingLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters);
void pcclTestGinPutSignalLatencyPingLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters);
void pcclTestGinPutCounterLatencyPingLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters);

void pcclTestGinGetLatencyPingLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters);

void pcclTestGinSignalLatencyPingLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters);

void pcclTestGinPutLatencyPingPongLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters);
void pcclTestGinPutSignalLatencyPingPongLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters);

void pcclTestGinSignalLatencyPingPongLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters);
