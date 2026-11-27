/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include "pccl.h"
#include "pccl_device.h"
#include "args.h"

// Extension headers declare SignalSet for PROXY even when native TROCE is disabled.
#if defined(PCCLX_GIN_TROCE_ENABLE)
#define PCCL_TEST_GIN_HAS_SIGNAL_SET 1
#else
#define PCCL_TEST_GIN_HAS_SIGNAL_SET 0
#endif

typedef struct {
  pcclDevComm dcomm;
  void* devBuf;
  pcclWindow_t devBufWindow;
  void* hostBuf;
  pcclWindow_t hostBufWindow;
} ginContext_t;

size_t pcclTestGinMaxBufferBytes(void);

void pcclTestGinDevCommCreate(pcclComm_t comm, const ginArgs_t* args, ginContext_t* ctx);
void pcclTestGinDevCommDestroy(pcclComm_t comm, ginContext_t* ctx);
