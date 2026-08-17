/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _BARRIER_H_
#define _BARRIER_H_

int barrierInit(int rank, int worldSize);
int barrier(int rank, int worldSize);
void barrierDeinit(int rank, int worldSize);

int allreduceFromSocket(double *value, int rank, int avg);

#endif
