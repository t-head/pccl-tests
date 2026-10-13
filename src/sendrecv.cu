/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "hggc_runtime.h"
#include "common.h"

void SendRecvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t SendRecvInitData(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    int seed = getSeed("SendRecv", rep, rank);
    HGGCCHECK(hggcMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, rank*sendcount, type, pcclSum, seed, 1, 0));
    int peer = (rank-1+nranks)%nranks;
    TESTCHECK(InitData(args->expected[i], recvcount, peer*recvcount, type, pcclSum, seed, 1, 0));
    HGGCCHECK(hggcDeviceSynchronize());
  }
  // We don't support in-place sendrecv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void SendRecvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

testResult_t SendRecvRunColl(void* sendbuff, void* recvbuff, size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclComm_t comm, hggcStream_t stream) {
  int nRanks;
  PCCLCHECK(pcclCommCount(comm, &nRanks));
  int rank;
  PCCLCHECK(pcclCommUserRank(comm, &rank));
  int recvPeer = (rank-1+nRanks) % nRanks;
  int sendPeer = (rank+1) % nRanks;

  PCCLCHECK(pcclGroupStart());
  PCCLCHECK(pcclSend(sendbuff, count, type, sendPeer, comm, stream));
  PCCLCHECK(pcclRecv(recvbuff, count, type, recvPeer, comm, stream));
  PCCLCHECK(pcclGroupEnd());
  return testSuccess;
}

struct testColl sendRecvTest = {
  "SendRecv",
  SendRecvGetCollByteCount,
  SendRecvInitData,
  SendRecvGetBw,
  SendRecvRunColl
};

void SendRecvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t *kerNum, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  SendRecvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, 1, nranks);
}

testResult_t SendRecvRunTest(struct threadArgs* args, int root, pcclDataType_t type, const char* typeName, pcclRedOp_t op, const char* opName) {
  args->collTest = &sendRecvTest;
  pcclDataType_t *run_types;
  pcclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if ((int)op != -1) {
    op_count = 1;
    run_ops = &op;
    run_opnames = &opName;
  } else {
    op_count = test_opnum;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=0; j<op_count; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

struct testEngine sendRecvEngine = {
  SendRecvGetBuffSize,
  SendRecvRunTest
};

#pragma weak pcclTestEngine=sendRecvEngine
