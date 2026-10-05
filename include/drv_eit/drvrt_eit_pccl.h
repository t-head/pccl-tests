/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef __HGGCRT_EIT_PCCL__
#define __HGGCRT_EIT_PCCL__

#include "hggc_uuid.h"
#include "hggcrt_eit_common.h"
#include "hggc_runtime_api.h"

HG_DEFINE_UUID(HGGC_EIT_ID_PCCL, 0xcdcddd2d, 0x6bf1, 0xe94e, 0x73ed, 0xb91882631679)

typedef struct HGGCeitPccl_st
{
  hggcError_t(HGGCAPI *hggcEitFuncSetAttribute)(const void* pFunc, enum hggcEitFuncAttribute attr, void* pData, uint32_t size);
  hggcError_t(HGGCAPI* hggcStreamCreateWithRingId)(hggcStream_t* pStream,
                                                   unsigned int flags,
                                                   enum hggcQueueType queueType,
                                                   unsigned int pipeIdx,
                                                   unsigned int dmaGroupIdx,
                                                   unsigned int dmaInnerIdx,
                                                   unsigned int icnRingIdx);
} HGGCeitPccl;

#endif // __HGGCRT_EIT_PCCL__
