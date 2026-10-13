/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "hggc_runtime.h"
#include "common.h"

#define ALIGN 4

void HyperCubeGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t base = (count/(ALIGN*nranks))*ALIGN;
  *sendcount = base;
  *recvcount = base*nranks;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

testResult_t HyperCubeInitData(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    int seed = getSeed("HyperCube", rep, rank);
    HGGCCHECK(hggcMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? ((char*)args->recvbuffs[i])+rank*args->sendBytes : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, pcclSum, seed, 1, 0));
    for (int j=0; j<nranks; j++) {
      int peerSeed = getSeed("HyperCube", rep, j);
      TESTCHECK(InitData((char*)args->expected[i] + args->sendBytes*j, sendcount, 0, type, pcclSum, peerSeed, 1, 0));
    }
    HGGCCHECK(hggcDeviceSynchronize());
  }
  return testSuccess;
}

void HyperCubeGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * (nranks - 1)) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

testResult_t HyperCubeRunColl(void* sendbuff, void* recvbuff, size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclComm_t comm, hggcStream_t stream) {
  char* sbuff = (char*)sendbuff;
  char* rbuff = (char*)recvbuff;
  int nRanks;
  PCCLCHECK(pcclCommCount(comm, &nRanks));
  int rank;
  PCCLCHECK(pcclCommUserRank(comm, &rank));
  size_t rankSize = count * wordSize(type);

  if (rbuff+rank*rankSize != sbuff) HGGCCHECK(hggcMemcpyAsync(rbuff+rank*rankSize, sbuff, rankSize, hggcMemcpyDeviceToDevice, stream));

  // Hypercube AllGather
  for (int mask=1; mask<nRanks; mask<<=1) {
    PCCLCHECK(pcclGroupStart());
    int s = rank & ~(mask-1);
    int r = s ^ mask;
    PCCLCHECK(pcclSend(rbuff+s*rankSize, count*mask, type, rank^mask, comm, stream));
    PCCLCHECK(pcclRecv(rbuff+r*rankSize, count*mask, type, rank^mask, comm, stream));
    PCCLCHECK(pcclGroupEnd());
  }
  return testSuccess;
}

struct testColl hyperCubeTest = {
  "HyperCube",
  HyperCubeGetCollByteCount,
  HyperCubeInitData,
  HyperCubeGetBw,
  HyperCubeRunColl
};

void HyperCubeGetBuffSize(size_t *sendcount, size_t *recvcount, size_t *kerNum, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  HyperCubeGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, 1, nranks);

  int power = 0;
  if (nranks > 1 && !(nranks & (nranks - 1))) {
    while (nranks > 1) {
      nranks >>= 1;
      power++;
    }
    *kerNum = power;
  }
}

testResult_t HyperCubeRunTest(struct threadArgs* args, int root, pcclDataType_t type, const char* typeName, pcclRedOp_t op, const char* opName) {
  args->collTest = &hyperCubeTest;
  pcclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  // Check if this is a power of 2
  int nRanks = args->nProcs*args->nThreads*args->nGpus;
  if (args->nGpus > 1 && nRanks != 2) {
    printf("hypercube don't support multiple devices in a single thread.\n");
    return testSuccess;
  }
  if (nRanks > 1 && !(nRanks & (nRanks - 1))) {
    for (int i=0; i<type_count; i++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (pcclRedOp_t)0, "null", -1));
    }
  } else {
    printf("nRanks %d is not a power of 2, skipping\n", nRanks);
  }

  return testSuccess;
}

struct testEngine hyperCubeEngine = {
  HyperCubeGetBuffSize,
  HyperCubeRunTest
};

#pragma weak pcclTestEngine=hyperCubeEngine
