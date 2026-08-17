/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef __HGGC_EIT_COMMON_TYPES__
#define __HGGC_EIT_COMMON_TYPES__
/// @brief EIT stream creation flags, super set of HGstream_flags.
enum HGeitStreamFlags {
  HG_EIT_STREAM_DEFAULT      = 0x0,        /**< Default stream flag */
  HG_EIT_STREAM_NON_BLOCKING = (1u <<  0), /**< Stream does not synchronize with stream 0 (the NULL stream) */
  HG_EIT_STREAM_COPY_ENGINE  = (1u << 16), /**< Stream prefers to performs memory processing on copy engine */
};
enum HGeitFuncAttribute {
  HG_EIT_FUNC_ATTRIBUTE_KERNEL_CONTROL   = 0,
  HG_EIT_FUNC_ATTRIBUTE_DISPATCH_CU_MASK = 1,
  HG_EIT_FUNC_ATTRIBUTE_DISPATCH_STRTEGY = 200,
};
static_assert(static_cast<uint32_t>(HG_STREAM_DEFAULT) == HG_EIT_STREAM_DEFAULT &&
              static_cast<uint32_t>(HG_STREAM_NON_BLOCKING) == HG_EIT_STREAM_NON_BLOCKING,
              "EIT stream flags should match the public stream flags!");
#endif // __HGGC_EIT_COMMON_TYPES__