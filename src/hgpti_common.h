/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _HGPTI_CALLBACK_H_
#define _HGPTI_CALLBACK_H_

#include "common.h"
#include "hgpti.h"

#define HGPTICHECK(cmd) do {                        \
  HGptiResult err = static_cast<HGptiResult>(cmd);  \
  if( err != HGPTI_SUCCESS ) {                      \
    char hostname[1024];                            \
    const char *errstr;                             \
    getHostName(hostname, 1024);                    \
    hgptiGetResultString(err, &errstr);             \
    printf("%s: Test HGPTI failure %s:%d '%s'\n",   \
         hostname,                                  \
        __FILE__,__LINE__,errstr);                  \
    return testHgptiError;                          \
  }                                                 \
} while(0)

testResult_t hgptiInit(int iters, int kerNum, int devNum);
testResult_t hgptiFree();

testResult_t hgptiClean();
testResult_t hgptiFlush(double *deltaSecPerGpusPerIter);

#endif
