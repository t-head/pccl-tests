/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "hggc_runtime.h"
#include "common.h"

void AllGatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count/nranks;
  *recvcount = (count/nranks)*nranks;
  *sendInplaceOffset = count/nranks;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t AllGatherInitData(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    int seed = getSeed("AllGather", rep, rank);
    HGGCCHECK(hggcMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? ((char*)args->recvbuffs[i])+rank*args->sendBytes : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, pcclSum, seed, 1, 0));
    for (int j=0; j<nranks; j++) {
      int peerSeed = getSeed("AllGather", rep, j);
      TESTCHECK(InitData((char*)args->expected[i] + args->sendBytes*j, sendcount, 0, type, pcclSum, peerSeed, 1, 0));
    }
  }
  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    HGGCCHECK(hggcDeviceSynchronize());
  }
  return testSuccess;
}

void AllGatherGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1))/((double)nranks);
  *busBw = baseBw * factor;
}

#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
testResult_t AllGatherRmaPut(void* sendWindow, size_t sendoffset, void* recvWindow, size_t recvoffset,
                             size_t count, pcclDataType_t type, int root, pcclComm_t comm, hggcStream_t stream) {
  (void)root;
  int rank, nranks;
  PCCLCHECK(pcclCommUserRank(comm, &rank));
  PCCLCHECK(pcclCommCount(comm, &nranks));

  pcclWindow_t sendWin = (pcclWindow_t)sendWindow;
  pcclWindow_t recvWin = (pcclWindow_t)recvWindow;

  void* sendPtr = NULL;
  void* recvPtr = NULL;
  PCCLCHECK(pcclWinGetUserPtr(comm, sendWin, &sendPtr));
  PCCLCHECK(pcclWinGetUserPtr(comm, recvWin, &recvPtr));

  size_t eltSize = wordSize(type);
  size_t bytes = count * eltSize;
  const int nctx = rmaCtxCount;

  bool isInPlace = ((char*)sendPtr + sendoffset == (char*)recvPtr + recvoffset + rank * bytes);
  size_t peerWinOffset = recvoffset + rank * bytes;

  pcclWaitSignalDesc_t* waitDescs = (pcclWaitSignalDesc_t*)malloc(sizeof(pcclWaitSignalDesc_t) * nranks);
  if (waitDescs == NULL) {
    return testInternalError;
  }

  int descIdx = 0;
  for (int i = 0; i < nranks; i++) {
    if (isInPlace && i == rank) {
      continue;
    }
    waitDescs[descIdx].opCnt = 1;
    waitDescs[descIdx].peer = i;
    waitDescs[descIdx].sigIdx = i % NUM_RMA_SIG;
    waitDescs[descIdx].ctx = (i + rank) % nctx;
    descIdx++;
  }

  PCCLCHECK(pcclGroupStart());
  for (int peer = 0; peer < nranks; peer++) {
    int targetRank = (rank + peer) % nranks;
    if (isInPlace && targetRank == rank) {
      continue;
    }
    PCCLCHECK(pcclPutSignal((char*)sendPtr + sendoffset, count, type, targetRank,
                      recvWin, peerWinOffset, rank % NUM_RMA_SIG, (rank + targetRank) % nctx, 0, comm, stream));
  }
  PCCLCHECK(pcclGroupEnd());

  // In-place single-rank runs skip every descriptor; pcclWaitSignal rejects nDesc == 0.
  if (descIdx > 0) {
    PCCLCHECK(pcclWaitSignal(descIdx, waitDescs, comm, stream));
  }
  free(waitDescs);
  return testSuccess;
}
#endif

testResult_t AllGatherRunColl(void* sendbuff, void* recvbuff, size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclComm_t comm, hggcStream_t stream) {
  PCCLCHECK(pcclAllGather(sendbuff, recvbuff, count, type, comm, stream));
  return testSuccess;
}

struct testColl allGatherTest = {
  "AllGather",
  AllGatherGetCollByteCount,
  AllGatherInitData,
  AllGatherGetBw,
  AllGatherRunColl
};

void AllGatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t *kerNum, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllGatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, 1, nranks);
}

testResult_t AllGatherRunTest(struct threadArgs* args, int root, pcclDataType_t type, const char* typeName, pcclRedOp_t op, const char* opName) {
  args->collTest = &allGatherTest;
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

  for (int i=0; i<type_count; i++) {
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (pcclRedOp_t)0, "null", -1));
  }
  return testSuccess;
}

struct testEngine allGatherEngine = {
  AllGatherGetBuffSize,
  AllGatherRunTest
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
  , NULL, NULL,
  AllGatherRmaPut
#endif
};

#pragma weak pcclTestEngine=allGatherEngine
