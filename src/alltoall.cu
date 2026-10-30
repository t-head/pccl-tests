/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "hggc_runtime.h"
#include "common.h"
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
#include "vector_types.h"
#endif

void AlltoAllGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = (count/nranks)*nranks;
  *recvcount = (count/nranks)*nranks;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = count/nranks;
}

testResult_t AlltoAllInitData(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    int seed = getSeed("AlltoAll", rep, rank);
    HGGCCHECK(hggcMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, pcclSum, seed, 1, 0));
    for (int j=0; j<nranks; j++) {
      size_t partcount = sendcount/nranks;
      int peerSeed = getSeed("AlltoAll", rep, j);
      TESTCHECK(InitData((char*)args->expected[i] + j*partcount*wordSize(type), partcount, rank*partcount, type, pcclSum, peerSeed, 1, 0));
    }
  }
  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    HGGCCHECK(hggcDeviceSynchronize());
  }
  // We don't support in-place alltoall
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
testResult_t AlltoAllGetDevCommRequirements(int deviceImpl, pcclDevCommRequirements* reqs, pcclComm_t comm) {
  if (!reqs || !comm) return testInternalError;

  pcclCommProperties_t commProperties = PCCL_COMM_PROPERTIES_INITIALIZER;
  if (pcclCommQueryProperties(comm, &commProperties) != pcclSuccess) {
    return testPcclError;
  }

  switch(deviceImpl) {
    case 1:
    case 2:
      if (commProperties.nRanks != pcclTeamLsa(comm).nRanks) {
        fprintf(stderr, "DeviceImplementation 1 and 2 requires P2P "
                        "connectivity across all ranks. Not all ranks of this "
                        "communicator have P2P connectivity.\n");
        return testInvalidUsage;
      }
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 3:
    case 4:
      if (commProperties.ginType == PCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not "
                        "enabled for this communicator.\n");
        return testInvalidUsage;
      }
      reqs->barrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      reqs->ginConnectionType = PCCL_GIN_CONNECTION_FULL;
      return testSuccess;
    default:
      fprintf(stderr, "ERROR: alltoall device implementation %d is not supported.\n", deviceImpl);
      return testNotImplemented;
  }
}

template <typename T>
__device__ void AlltoAllScalarImpl(pcclWindow_t sendwin, size_t sendoffset, pcclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int tid, int nthreads) {
  T* sendPtr = (T*)pcclGetLsaPointer(sendwin, sendoffset, rank);

  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int peer = 0; peer < nRanks; peer++) {
      T value = sendPtr[peer * count + offset];
      T* recvPtr = (T*)pcclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[rank * count + offset] = value;
    }
  }
}

template <typename T>
__global__ void IcnAlltoAllKernel(pcclWindow_t sendwin, size_t sendoffset, pcclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct pcclDevComm devComm) {
  pcclLsaBarrierSession<pcclCoopCta> bar { pcclCoopCta(), devComm, pcclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(pcclCoopCta(), hggc::memory_order_acquire);

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);

  bar.sync(pcclCoopCta(), hggc::memory_order_release);
}

template <typename T>
__global__ void IcnAlltoAllKernelOptimized(pcclWindow_t sendwin, size_t sendoffset, pcclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct pcclDevComm devComm) {
  pcclLsaBarrierSession<pcclCoopCta> bar { pcclCoopCta(), devComm, pcclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(pcclCoopCta(), hggc::memory_order_acquire);

  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN) / sizeof(T);
  constexpr int UNROLL_FACTOR = 128/sizeof(TN);
  constexpr int PEER_UNROLL = 2;

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  T* sendPtr = (T*)pcclGetLsaPointer(sendwin, sendoffset, rank);

  bool canVectorize = (sizeof(TN) > sizeof(T)) &&
                      (reinterpret_cast<uintptr_t>(sendPtr) % sizeof(TN) == 0) &&
                      ((count * sizeof(T)) % sizeof(TN) == 0);

  if (canVectorize) {
    size_t vector_count = count / VECTOR_FACTOR;
    int elements_per_iteration = nthreads * UNROLL_FACTOR;

    size_t aligned_vector_count = (vector_count / elements_per_iteration) * elements_per_iteration;
    for (size_t base_offset = tid; base_offset < aligned_vector_count; base_offset += elements_per_iteration) {
      for (int peerBase = 0; peerBase < nRanks; peerBase += PEER_UNROLL) {
        int peersInGroup = min(PEER_UNROLL, nRanks - peerBase);

        #pragma unroll
        for (int p = 0; p < peersInGroup; p++) {
          int peer = peerBase + p;
          TN* sendVecPtr = (TN*)(sendPtr + peer * count);
          TN* recvVecPtr = (TN*)((T*)pcclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
          TN values[UNROLL_FACTOR];

          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++) {
            size_t offset = base_offset + i * nthreads;
            values[i] = sendVecPtr[offset];
          }
          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++) {
            size_t offset = base_offset + i * nthreads;
            recvVecPtr[offset] = values[i];
          }
        }
      }
    }

    for (size_t base_offset = aligned_vector_count + tid; base_offset < vector_count; base_offset += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        TN* sendVecPtr = (TN*)(sendPtr + peer * count);
        TN* recvVecPtr = (TN*)((T*)pcclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
        recvVecPtr[base_offset] = sendVecPtr[base_offset];
      }
    }

    size_t scalar_start = vector_count * VECTOR_FACTOR;
    for (size_t offset = scalar_start + tid; offset < count; offset += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        T value = sendPtr[peer * count + offset];
        T* recvPtr = (T*)pcclGetLsaPointer(recvwin, recvoffset, peer);
        recvPtr[rank * count + offset] = value;
      }
    }
  } else {
    AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);
  }

  bar.sync(pcclCoopCta(), hggc::memory_order_release);
}

#ifdef PCCLX_HAS_LSA_BULK
constexpr int WARP_SIZE = 32;
constexpr int BULK_UNROLL = 4;
constexpr int BULK_MIN_CTAS = 16; // Minimum CTAs to overlap remote-store latency.
constexpr size_t BULK_CHUNK_BYTES = BULK_UNROLL * PCCLX_LSA_BULK_WARP_BYTES;

template <typename T>
__global__ void IcnAlltoAllBulkKernel(pcclWindow_t sendwin, size_t sendoffset, pcclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct pcclDevComm devComm) {
  pcclLsaBarrierSession<pcclCoopCta> bar { pcclCoopCta(), devComm, pcclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.arrive(pcclCoopCta(), hggc::memory_order_relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;
  T* sendLocal = (T*)pcclGetLocalPointer(sendwin, sendoffset);
  T* recvLocal = (T*)pcclGetLocalPointer(recvwin, recvoffset);
  const size_t peerBytes = count * sizeof(T);
  const int warpsPerBlock = blockDim.x / WARP_SIZE;
  const int totalWarps = gridDim.x * warpsPerBlock;
  const bool useBulk = totalWarps >= nRanks &&
                       reinterpret_cast<uintptr_t>(sendLocal) % 128 == 0 &&
                       reinterpret_cast<uintptr_t>(recvLocal) % 128 == 0 &&
                       peerBytes % PCCLX_LSA_BULK_WARP_BYTES == 0;

  if (!useBulk) {
    bar.wait(pcclCoopCta(), hggc::memory_order_relaxed);
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);
    bar.arrive(pcclCoopCta(), hggc::memory_order_release);
    bar.wait(pcclCoopCta(), hggc::memory_order_relaxed);
    return;
  }

  const int warpLinear = blockIdx.x * warpsPerBlock + threadIdx.x / WARP_SIZE;
  const int peerSlot = warpLinear % nRanks;
  int dataWarp = warpLinear / nRanks;
  const int warpsForPeer = (totalWarps - 1 - peerSlot) / nRanks + 1;
  const int peer = (rank + peerSlot) % nRanks;
  const size_t bulkBlockCount = peerBytes / PCCLX_LSA_BULK_WARP_BYTES;
  char* src = (char*)sendLocal + (size_t)peer * peerBytes + (size_t)dataWarp * BULK_CHUNK_BYTES;
  char* dst = (char*)pcclGetLsaPointer(recvwin, recvoffset, peer) + (size_t)rank * peerBytes + (size_t)dataWarp * BULK_CHUNK_BYTES;
  int64_t iters = (int64_t)((peerBytes + BULK_CHUNK_BYTES - 1) / BULK_CHUNK_BYTES) - dataWarp;

  uint4 values[BULK_UNROLL];
  if (iters > 0) {
#pragma unroll
    for (int u = 0; u < BULK_UNROLL; u++) {
      if ((size_t)dataWarp * BULK_UNROLL + u < bulkBlockCount)
        values[u] = pcclxLsaBulkWarpLoad<PCCLX_LSA_BULK_LD_CG>(src + u * PCCLX_LSA_BULK_WARP_BYTES);
    }
  }
  bar.wait(pcclCoopCta(), hggc::memory_order_relaxed);

  while (iters > 0) {
#pragma unroll
    for (int u = 0; u < BULK_UNROLL; u++) {
      if ((size_t)dataWarp * BULK_UNROLL + u < bulkBlockCount) {
        if (peer == rank) pcclxLsaBulkWarpStore<PCCLX_LSA_BULK_ST_CG>(dst + u * PCCLX_LSA_BULK_WARP_BYTES, values[u]);
        else pcclxLsaBulkWarpPeerStore<PCCLX_LSA_BULK_RMT_C, PCCLX_LSA_BULK_RTMD_PA>(dst + u * PCCLX_LSA_BULK_WARP_BYTES, values[u]);
      }
    }
    src += (size_t)warpsForPeer * BULK_CHUNK_BYTES;
    dst += (size_t)warpsForPeer * BULK_CHUNK_BYTES;
    dataWarp += warpsForPeer;
    iters -= warpsForPeer;
    if (iters > 0) {
#pragma unroll
      for (int u = 0; u < BULK_UNROLL; u++) {
        if ((size_t)dataWarp * BULK_UNROLL + u < bulkBlockCount)
          values[u] = pcclxLsaBulkWarpLoad<PCCLX_LSA_BULK_LD_CG>(src + u * PCCLX_LSA_BULK_WARP_BYTES);
      }
    }
  }
  bar.arrive(pcclCoopCta(), hggc::memory_order_release);
  bar.wait(pcclCoopCta(), hggc::memory_order_relaxed);
}
#endif

template <typename T>
__global__ void GinAlltoAllKernel(pcclWindow_t sendwin, size_t sendoffset, pcclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct pcclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = blockIdx.x;
  pcclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  pcclBarrierSession<pcclCoopCta> bar { pcclCoopCta(), pcclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(pcclCoopCta(), hggc::memory_order_acquire, pcclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  const size_t size = count * sizeof(T);
  for (int r=tid; r<devComm.nRanks; r+=nthreads) {
    gin.put(pcclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, pcclGin_SignalInc{signalIndex});
  }

  int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(pcclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(pcclCoopCta());

  bar.sync(pcclCoopCta(), hggc::memory_order_release, pcclGinFenceLevel::Relaxed);
}

template <typename T>
__global__ void HybridAlltoAllKernel(pcclWindow_t sendwin, size_t sendoffset, pcclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct pcclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = blockIdx.x;
  pcclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  pcclBarrierSession<pcclCoopCta> bar { pcclCoopCta(), pcclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(pcclCoopCta(), hggc::memory_order_acquire, pcclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  pcclTeam world = pcclTeamWorld(devComm);
  pcclTeam lsa = pcclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int lsaSize  = lsa.nRanks;

  const size_t size = count * sizeof(T);
  for (int r = tid; r < world.nRanks; r += nthreads) {
    if (r < startLsa || r >= startLsa + lsaSize) {
      gin.put(world, r,
          recvwin, recvoffset + world.rank * size,
          sendwin, sendoffset + r * size,
          size, pcclGin_SignalInc{signalIndex});
    }
  }

  T* sendLocal = (T*)pcclGetLocalPointer(sendwin, sendoffset);
  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int lp = 0; lp < lsa.nRanks; lp++) {
      int wr = startLsa + lp;
      T* recvPtr = (T*)pcclGetLsaPointer(recvwin, recvoffset, lp);
      recvPtr[world.rank * count + offset] = sendLocal[wr * count + offset];
    }
  }

  int numRemotePeers = world.nRanks - lsa.nRanks;
  int receivingCta = (world.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(pcclCoopCta(), signalIndex, signalValue + numRemotePeers);
  gin.flush(pcclCoopCta());

  bar.sync(pcclCoopCta(), hggc::memory_order_release, pcclGinFenceLevel::Relaxed);
}

testResult_t AlltoAllRunCollDevice(void* sendwin, size_t sendoffset, void* recvwin, size_t recvoffset,
    size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclDevComm* devComm, hggcStream_t stream, int deviceImpl) {
  switch(deviceImpl) {
    case 1:
      TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(IcnAlltoAllKernel, type, op),
                 sendwin, sendoffset, recvwin, recvoffset, count, type, op, root, devComm, stream));
      return testSuccess;
    case 2:
#ifdef PCCLX_HAS_LSA_BULK
      if (deviceCtaCount >= BULK_MIN_CTAS && count * wordSize(type) % PCCLX_LSA_BULK_WARP_BYTES == 0 &&
          sendoffset % 128 == 0 && recvoffset % 128 == 0) {
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(IcnAlltoAllBulkKernel, type, op),
                   sendwin, sendoffset, recvwin, recvoffset, count, type, op, root, devComm, stream));
        return testSuccess;
      }
#endif
      TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(IcnAlltoAllKernelOptimized, type, op),
                 sendwin, sendoffset, recvwin, recvoffset, count, type, op, root, devComm, stream));
      return testSuccess;
    case 3:
      TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(GinAlltoAllKernel, type, op),
                 sendwin, sendoffset, recvwin, recvoffset, count, type, op, root, devComm, stream));
      return testSuccess;
    case 4:
      TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(HybridAlltoAllKernel, type, op),
                 sendwin, sendoffset, recvwin, recvoffset, count, type, op, root, devComm, stream));
      return testSuccess;
    default:
      fprintf(stderr, "ERROR: alltoall device implementation %d is not supported.\n", deviceImpl);
      return testNotImplemented;
  }
}
#endif

#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
testResult_t AlltoAllRmaPut(void* sendWindow, size_t sendoffset, void* recvWindow, size_t recvoffset,
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
  size_t chunkBytes = count * eltSize;
  const int nctx = rmaCtxCount;

  pcclWaitSignalDesc_t* waitDescs = (pcclWaitSignalDesc_t*)malloc(sizeof(pcclWaitSignalDesc_t) * nranks);
  if (waitDescs == NULL) {
    return testInternalError;
  }

  for (int i = 0; i < nranks; i++) {
    waitDescs[i].opCnt = 1;
    waitDescs[i].peer = i;
    waitDescs[i].sigIdx = i % NUM_RMA_SIG;
    waitDescs[i].ctx = (i + rank) % nctx;
  }

  PCCLCHECK(pcclGroupStart());
  for (int peer = 0; peer < nranks; peer++) {
    int targetRank = (rank + peer) % nranks;
    void* srcPtr = (char*)sendPtr + sendoffset + targetRank * chunkBytes;
    size_t dstOffset = recvoffset + rank * chunkBytes;

    PCCLCHECK(pcclPutSignal(srcPtr, count, type, targetRank,
                      recvWin, dstOffset, rank % NUM_RMA_SIG, (rank + targetRank) % nctx, 0, comm, stream));
  }
  PCCLCHECK(pcclGroupEnd());

  PCCLCHECK(pcclWaitSignal(nranks, waitDescs, comm, stream));
  free(waitDescs);
  return testSuccess;
}
#endif

testResult_t AlltoAllRunColl(void* sendbuff, void* recvbuff, size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclComm_t comm, hggcStream_t stream) {
  if (count == 0) return testSuccess;
  char* sptr = (char*)sendbuff;
  char* rptr = (char*)recvbuff;
  PCCLCHECK(pcclAlltoAll(sptr, rptr, count, type, comm, stream));
  return testSuccess;
}

struct testColl alltoAllTest = {
  "AlltoAll",
  AlltoAllGetCollByteCount,
  AlltoAllInitData,
  AlltoAllGetBw,
  AlltoAllRunColl
};

void AlltoAllGetBuffSize(size_t *sendcount, size_t *recvcount, size_t *kerNum, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, 1, nranks);
}

testResult_t AlltoAllRunTest(struct threadArgs* args, int root, pcclDataType_t type, const char* typeName, pcclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllTest;
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

struct testEngine alltoAllEngine = {
  AlltoAllGetBuffSize,
  AlltoAllRunTest,
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
  AlltoAllGetDevCommRequirements,
  AlltoAllRunCollDevice,
  AlltoAllRmaPut
#endif
};

#pragma weak pcclTestEngine=alltoAllEngine
