/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef __HGGCRT_EIT_COMMON__
#define __HGGCRT_EIT_COMMON__

/**
 * HGGC eit function attributes that can be set using ::hggcEitFuncSetAttribute
 */
enum hggcEitFuncAttribute
{
    /**
     * attributes the same as hgEitFuncAttribute parameters
     * HGeitFuncAttribute.
    */
    hggcEitFuncAttributeKernelControl  = 0, /**< Kernel control setting */
    hggcEitFuncAttributeDispatchCuMask = 1, /**< Kernel dispatch cu mask setting */

    /**
     * hggcEitFuncSetAttribute special attributes.
    */
    hggcEitFuncAttributeDispatchStrategy = 200, /**< Kernel dispatch strategy setting */
    hggcEitFuncAttributeMax
};

typedef struct hggcDispatchCuMask_st
{
    uint64_t cuMask;
    uint64_t cuMask1;
    uint64_t cuMask2;
} hggcDispatchCuMask;

enum hggcQueueType {
    hggcQueueTypeNone    = 0,
    hggcQueueTypeCompute = 1 << 0,
    hggcQueueTypeDma     = 1 << 1,
    hggcQueueTypeIcn     = 1 << 2
};

#endif // __HGGCRT_EIT_COMMON__
