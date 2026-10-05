/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef __HGGC_UUID_H__
#define __HGGC_UUID_H__

#include "hggc.h"

#define HG_UUID_CHAR(x) (char)((unsigned char)((x) & 0xff))

#define HG_DEFINE_UUID(var, timeLow, timeMid, timeHigh, varSeq, node) \
  static const HGuuid var =                                           \
  {                                                                   \
    {                                                                 \
      HG_UUID_CHAR((timeLow)  >> 24), HG_UUID_CHAR((timeLow)  >> 16), \
      HG_UUID_CHAR((timeLow)  >> 8 ), HG_UUID_CHAR((timeLow)  >> 0 ), \
      HG_UUID_CHAR((timeMid)  >> 8 ), HG_UUID_CHAR((timeMid)  >> 0 ), \
      HG_UUID_CHAR((timeHigh) >> 8 ), HG_UUID_CHAR((timeHigh) >> 0 ), \
      HG_UUID_CHAR((varSeq)   >> 8 ), HG_UUID_CHAR((varSeq)   >> 0 ), \
      HG_UUID_CHAR((node)     >> 40), HG_UUID_CHAR((node)     >> 32), \
      HG_UUID_CHAR((node)     >> 24), HG_UUID_CHAR((node)     >> 16), \
      HG_UUID_CHAR((node)     >> 8 ), HG_UUID_CHAR((node)     >> 0 ), \
    }                                                                 \
  };

/**
 ***********************************************************************************************************************
 * @brief  UUID 48-bit node sequence.
 *
 ***********************************************************************************************************************
 */
typedef struct _UuidNode {
  uint8_t raw[6];
} UuidNode;

/**
 ***********************************************************************************************************************
 * @brief UUID data storage structure
 *
 * @note Byte order of data is determined by Variant:
 *       * `Variant::Rfc4122` denotes network byte order (big-endian)
 *       * `Variant::MsCompatible` denotes host byte order (mixed-endian)
 *
 * @note `variantAndSequence` and `node` are always network byte order
 ***********************************************************************************************************************
 */
typedef struct _UuidData {
  uint32_t timeLow;            /// low 32-bits of timestamp
  uint16_t timeMid;            /// middle 16-bits of timestamp
  uint16_t timeHighAndVersion; /// 4-bit version and high 12-bits of timestamp
  uint16_t variantAndSequence; /// 1 to 3-bit variant and 13 to 15-bit sequence id
  UuidNode node;               /// 48-bit node ID
} UuidData;

/// @brief An accessor union of UUID in both raw bytes and UuidData formats
union Uuid {
  uint8_t  raw[sizeof(UuidData)]; /// raw digits of the UUID as byte array
  uint64_t raw64[2];              /// raw data as two 64-bit registers
  UuidData data;                  /// helper accessor for UUID components
};


#endif // __HGGC_UUID_H__
