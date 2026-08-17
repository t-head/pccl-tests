/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "hggc_runtime.h"
#include "common.h"

void GatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = count/nranks;
  *recvcount = (count/nranks)*nranks;
  *sendInplaceOffset = count/nranks;
  *recvInplaceOffset = 0;
  *paramcount = count/nranks;
}

testResult_t GatherInitData(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    int seed = getSeed("Gather", rep, rank);
    HGGCCHECK(hggcMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? ((char*)args->recvbuffs[i])+rank*args->sendBytes : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, rank*sendcount, type, pcclSum, seed, 1, 0));
    HGGCCHECK(hggcMemcpy(args->expected[i], args->recvbuffs[i], args->expectedBytes, hggcMemcpyDefault));
    if (rank == root) {
      TESTCHECK(InitData(args->expected[i], nranks*sendcount, 0, type, pcclSum, seed, 1, 0));
    }
  }
  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    HGGCCHECK(hggcDeviceSynchronize());
  }
  return testSuccess;
}

void GatherGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t GatherRunColl(void* sendbuff, void* recvbuff, size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclComm_t comm, hggcStream_t stream) {
  if (count == 0) return testSuccess;

  char* sptr = (char*)sendbuff;
  char* rptr = (char*)recvbuff;
  PCCLCHECK(pcclGather(sptr, rptr, count, type, root, comm, stream));

  return testSuccess;
}

struct testColl gatherTest = {
  "Gather",
  GatherGetCollByteCount,
  GatherInitData,
  GatherGetBw,
  GatherRunColl
};

void GatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t *kerNum, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  GatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t GatherRunTest(struct threadArgs* args, int root, pcclDataType_t type, const char* typeName, pcclRedOp_t op, const char* opName) {
  args->collTest = &gatherTest;
  pcclDataType_t *run_types;
  const char **run_typenames;
  int type_count;
  int begin_root, end_root;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if (root != -1) {
    begin_root = end_root = root;
  } else {
    begin_root = 0;
    end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=begin_root; j<=end_root; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (pcclRedOp_t)0, "null", j));
    }
  }
  return testSuccess;
}

struct testEngine gatherEngine = {
  GatherGetBuffSize,
  GatherRunTest
};

#pragma weak pcclTestEngine=gatherEngine
