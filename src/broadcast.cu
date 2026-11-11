/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "hggc_runtime.h"
#include "common.h"

void BroadcastGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t BroadcastInitData(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);

  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    int seed = getSeed("Broadcast", rep, rank);
    HGGCCHECK(hggcMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    if (rank == root) TESTCHECK(InitData(data, sendcount, 0, type, pcclSum, seed, 1, 0));
    TESTCHECK(InitData(args->expected[i], recvcount, 0, type, pcclSum, seed, 1, 0));
  }
  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    HGGCCHECK(hggcDeviceSynchronize());
  }
  return testSuccess;
}

void BroadcastGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
testResult_t BroadcastRmaPut(void* sendWindow, size_t sendoffset, void* recvWindow, size_t recvoffset,
                             size_t count, pcclDataType_t type, int root, pcclComm_t comm, hggcStream_t stream) {
  int rank, nranks;
  PCCLCHECK(pcclCommUserRank(comm, &rank));
  PCCLCHECK(pcclCommCount(comm, &nranks));

  pcclWindow_t sendWin = (pcclWindow_t)sendWindow;
  pcclWindow_t recvWin = (pcclWindow_t)recvWindow;

  void* sendPtr = NULL;
  void* recvPtr = NULL;
  PCCLCHECK(pcclWinGetUserPtr(comm, sendWin, &sendPtr));
  PCCLCHECK(pcclWinGetUserPtr(comm, recvWin, &recvPtr));

  const int nctx = rmaCtxCount;

  PCCLCHECK(pcclGroupStart());
  if (rank == root) {
    for (int peer = 0; peer < nranks; peer++) {
      PCCLCHECK(pcclPutSignal((char*)sendPtr + sendoffset, count, type, peer,
                        recvWin, recvoffset, peer % NUM_RMA_SIG, (rank + peer) % nctx, 0, comm, stream));
    }
  }
  PCCLCHECK(pcclGroupEnd());

  pcclWaitSignalDesc_t waitDesc = {1, root, rank % NUM_RMA_SIG, (root + rank) % nctx};
  PCCLCHECK(pcclWaitSignal(1, &waitDesc, comm, stream));
  return testSuccess;
}
#endif

testResult_t BroadcastRunColl(void* sendbuff, void* recvbuff, size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclComm_t comm, hggcStream_t stream) {
  int rank;
  PCCLCHECK(pcclCommUserRank(comm, &rank));

  PCCLCHECK(pcclBroadcast(sendbuff, recvbuff, count, type, root, comm, stream));
  return testSuccess;
}

struct testColl broadcastTest = {
  "Broadcast",
  BroadcastGetCollByteCount,
  BroadcastInitData,
  BroadcastGetBw,
  BroadcastRunColl
};

void BroadcastGetBuffSize(size_t *sendcount, size_t *recvcount, size_t *kerNum, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  BroadcastGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, 1, nranks);
}

testResult_t BroadcastRunTest(struct threadArgs* args, int root, pcclDataType_t type, const char* typeName, pcclRedOp_t op, const char* opName) {
  args->collTest = &broadcastTest;
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

struct testEngine broadcastEngine = {
  BroadcastGetBuffSize,
  BroadcastRunTest
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
  , NULL, NULL,
  BroadcastRmaPut
#endif
};

#pragma weak pcclTestEngine=broadcastEngine
