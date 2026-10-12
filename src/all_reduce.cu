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

constexpr int WARP_SIZE = 32;
#endif

void AllReduceGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t AllReduceInitData(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    int seed = getSeed("AllReduce", rep, rank);
    HGGCCHECK(hggcMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, op, seed, nranks, rank));
    TESTCHECK(InitDataReduce(args->expected[i], recvcount, 0, type, op, seed, nranks));
  }
  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    HGGCCHECK(hggcDeviceSynchronize());
  }
  return testSuccess;
}

void AllReduceGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(2*(nranks - 1)))/((double)nranks);
  *busBw = baseBw * factor;
}

#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
testResult_t AllReduceGetDevCommRequirements(int deviceImpl, pcclDevCommRequirements* reqs, pcclComm_t comm) {
  if (!reqs || !comm) return testInternalError;

  pcclCommProperties_t commProperties = PCCL_COMM_PROPERTIES_INITIALIZER;
  if (pcclCommQueryProperties(comm, &commProperties) != pcclSuccess) {
    return testPcclError;
  }

  if (deviceImpl > 0 && commProperties.nRanks != pcclTeamLsa(comm).nRanks) {
    fprintf(stderr, "DeviceImplementation > 0 requires P2P connectivity across all "
                    "ranks. Not all ranks of this communicator have P2P connectivity.\n");
    return testInvalidUsage;
  }

  switch(deviceImpl) {
    case 1:
    case 2:
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 3:
    case 4:
      fprintf(stderr, "ERROR: all_reduce device implementation 3/4 (multimem kernels) "
                      "are not supported on PPU.\n");
      return testNotImplemented;
    default:
      fprintf(stderr, "ERROR: all_reduce device implementation %d is not supported.\n", deviceImpl);
      return testNotImplemented;
  }
}

template <typename T>
__global__ void allReduceLsaKernel(pcclWindow_t sendwin, size_t sendoffset, pcclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct pcclDevComm devComm) {
  pcclLsaBarrierSession<pcclCoopCta> bar { pcclCoopCta(), devComm, pcclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(pcclCoopCta(), hggc::memory_order_acquire);

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    T v = T{0};
    for (int peer=0; peer<nRanks; peer++) {
      T* sendPtr = (T*)pcclGetLsaPointer(sendwin, sendoffset, peer);
      v += sendPtr[offset];
    }
    for (int peer=0; peer<nRanks; peer++) {
      T* recvPtr = (T*)pcclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[offset] = v;
    }
  }
  bar.sync(pcclCoopCta(), hggc::memory_order_release);
}

template <typename T>
__global__ void allReduceLsaVectorizedKernel(pcclWindow_t sendwin, size_t sendoffset, pcclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct pcclDevComm devComm) {
  pcclLsaBarrierSession<pcclCoopCta> bar { pcclCoopCta(), devComm, pcclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(pcclCoopCta(), hggc::memory_order_acquire);

  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN)/sizeof(T);
  constexpr int UNROLL_FACTOR = 128/sizeof(TN);

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  const size_t alignment_offset = (sendoffset % sizeof(TN)) / sizeof(T);
  const size_t aligned_count = count - alignment_offset;
  const size_t vector_count = aligned_count / VECTOR_FACTOR;
  const size_t remainder = aligned_count % VECTOR_FACTOR;

  const int elements_per_block = globalNthreads * UNROLL_FACTOR;
  const int num_blocks = vector_count / elements_per_block;

  const int warp_id = globalTid / WARP_SIZE;
  const int lane_id = globalTid % WARP_SIZE;

  const int warp_offset = warp_id * WARP_SIZE * UNROLL_FACTOR;
  const int lane_offset = lane_id;
  const int warp_lane_offset = warp_offset + lane_offset;

  if (alignment_offset > 0) {
    for (size_t offset = globalTid; offset < alignment_offset; offset += globalNthreads) {
      T v_scalar = T{0};
      for (int peer=0; peer<nRanks; peer++) {
        T* remotePtr = (T*)pcclGetLsaPointer(sendwin, sendoffset, peer);
        v_scalar += remotePtr[offset];
      }
      for (int peer=0; peer<nRanks; peer++) {
        T* remotePtr = (T*)pcclGetLsaPointer(recvwin, recvoffset, peer);
        remotePtr[offset] = v_scalar;
      }
    }
  }

  for (int block = 0; block < num_blocks; block += 1) {
    TN v[UNROLL_FACTOR] = {TN{0}};
    const size_t block_offset = block * globalNthreads * UNROLL_FACTOR;
    for (int peer=0; peer<nRanks; peer++) {
#pragma unroll
      for (int i=0; i < UNROLL_FACTOR; i++) {
        const int stride_offset = i * WARP_SIZE;
        const size_t offset = warp_lane_offset + block_offset + stride_offset;
        TN* remotePtr = (TN*)pcclGetLsaPointer(sendwin, sendoffset + alignment_offset * sizeof(T), peer);
        v[i] = vectorAdd(v[i], remotePtr[offset]);
      }
    }
    for (int peer=0; peer<nRanks; ++peer) {
#pragma unroll
      for (int i=0; i < UNROLL_FACTOR; i++) {
        const int stride_offset = i * WARP_SIZE;
        const size_t offset = warp_lane_offset + block_offset + stride_offset;
        TN* remotePtr = (TN*)pcclGetLsaPointer(recvwin, recvoffset + alignment_offset * sizeof(T), peer);
        remotePtr[offset] = v[i];
      }
    }
  }

  const int block = num_blocks;
  TN v[UNROLL_FACTOR] = {TN{0}};
  const size_t block_offset = block * globalNthreads * UNROLL_FACTOR;
  for (int peer=0; peer<nRanks; peer++) {
#pragma unroll
      for (int i=0; i < UNROLL_FACTOR; i++) {
        const int stride_offset = i * WARP_SIZE;
        const size_t offset = warp_lane_offset + block_offset + stride_offset;
        if (offset < vector_count) {
          TN* remotePtr = (TN*)pcclGetLsaPointer(sendwin, sendoffset + alignment_offset * sizeof(T), peer);
          v[i] = vectorAdd(v[i], remotePtr[offset]);
        }
      }
  }
  for (int peer=0; peer<nRanks; ++peer) {
#pragma unroll
      for(int i=0; i < UNROLL_FACTOR; i++){
        const int stride_offset = i * WARP_SIZE;
        const size_t offset = warp_lane_offset + block_offset + stride_offset;
        if (offset < vector_count) {
          TN* remotePtr = (TN*)pcclGetLsaPointer(recvwin, recvoffset + alignment_offset * sizeof(T), peer);
          remotePtr[offset] = v[i];
        }
      }
  }

  if (remainder > 0) {
    const size_t remainder_start = alignment_offset + vector_count * VECTOR_FACTOR;

    for (size_t offset = globalTid; offset < remainder; offset += globalNthreads) {
      T v_scalar = T{0};
      const size_t actual_offset = remainder_start + offset;
      for (int peer=0; peer<nRanks; peer++) {
        T* remotePtr = (T*)pcclGetLsaPointer(sendwin, sendoffset, peer);
        v_scalar += remotePtr[actual_offset];
      }
      for (int peer=0; peer<nRanks; peer++) {
        T* remotePtr = (T*)pcclGetLsaPointer(recvwin, recvoffset, peer);
        remotePtr[actual_offset] = v_scalar;
      }
    }
  }

  bar.sync(pcclCoopCta(), hggc::memory_order_release);
}

#ifdef PCCLX_HAS_LSA_BULK
constexpr int BULK_UNROLL = 4;
constexpr size_t BULK_CHUNK_BYTES = BULK_UNROLL * PCCLX_LSA_BULK_WARP_BYTES;

template <typename T>
__global__ void allReduceLsaBulkKernel(pcclWindow_t sendwin, size_t sendoffset, pcclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct pcclDevComm devComm) {
  pcclLsaBarrierSession<pcclCoopCta> bar { pcclCoopCta(), devComm, pcclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.arrive(pcclCoopCta(), hggc::memory_order_relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;
  T* sendLocal = (T*)pcclGetLocalPointer(sendwin, sendoffset);
  T* recvLocal = (T*)pcclGetLocalPointer(recvwin, recvoffset);
  const size_t nBytes = count * sizeof(T);
  constexpr int UnrollPeers = 4;
  constexpr int FragElts = sizeof(uint4) / sizeof(T);
  const bool useBulk = nRanks > 1 && sizeof(T) == 4 &&
                       reinterpret_cast<uintptr_t>(sendLocal) % 128 == 0 &&
                       reinterpret_cast<uintptr_t>(recvLocal) % 128 == 0 &&
                       nBytes % BULK_CHUNK_BYTES == 0;

  if (!useBulk) {
    bar.wait(pcclCoopCta(), hggc::memory_order_relaxed);
    const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
    const int globalNthreads = blockDim.x * gridDim.x * nRanks;
    for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
      T value = T{0};
      for (int peer = 0; peer < nRanks; peer++) {
        T* src = (T*)pcclGetLsaPointer(sendwin, sendoffset, peer);
        value += src[offset];
      }
      for (int peer = 0; peer < nRanks; peer++) {
        T* dst = (T*)pcclGetLsaPointer(recvwin, recvoffset, peer);
        dst[offset] = value;
      }
    }
    bar.arrive(pcclCoopCta(), hggc::memory_order_release);
    bar.wait(pcclCoopCta(), hggc::memory_order_relaxed);
    return;
  }

  const int gw = pcclCoopBcast(pcclCoopWarp(), rank + nRanks * (blockIdx.x + gridDim.x * (threadIdx.x / WARP_SIZE)), 0);
  const int gwn = nRanks * gridDim.x * (blockDim.x / WARP_SIZE);
  char* sendBase = pcclCoopBcast(pcclCoopWarp(), (char*)pcclGetLsaPointer(sendwin, sendoffset, 0), 0);
  char* recvBase = pcclCoopBcast(pcclCoopWarp(), (char*)pcclGetLsaPointer(recvwin, recvoffset, 0), 0);
  const size_t sendPeerStride = (uintptr_t)pcclGetLsaPointer(sendwin, sendoffset, 1) - (uintptr_t)sendBase;
  const size_t recvPeerStride = (uintptr_t)pcclGetLsaPointer(recvwin, recvoffset, 1) - (uintptr_t)recvBase;
  char* localSend = sendBase + (size_t)rank * sendPeerStride;
  char* localRecv = recvBase + (size_t)rank * recvPeerStride;
  size_t chunkOffset = (size_t)gw * BULK_CHUNK_BYTES;
  int64_t iters = (int64_t)(nBytes / BULK_CHUNK_BYTES) - gw;

  uint4 local[BULK_UNROLL];
  if (iters > 0) {
#pragma unroll
    for (int u = 0; u < BULK_UNROLL; u++)
      local[u] = pcclxLsaBulkWarpLoad<PCCLX_LSA_BULK_LD_CG>(localSend + chunkOffset + u * PCCLX_LSA_BULK_WARP_BYTES);
  }
  bar.wait(pcclCoopCta(), hggc::memory_order_relaxed);

  while (iters > 0) {
    uint4 accum[BULK_UNROLL];
    const int firstPeer = (rank + 1) % nRanks;
    char* firstPeerBase = sendBase + (size_t)firstPeer * sendPeerStride + chunkOffset;
#pragma unroll
    for (int u = 0; u < BULK_UNROLL; u++) {
      uint4 remote = pcclxLsaBulkWarpPeerLoad<PCCLX_LSA_BULK_RMT_C, PCCLX_LSA_BULK_RTMD_PA>(firstPeerBase + u * PCCLX_LSA_BULK_WARP_BYTES);
      T* a = (T*)&local[u]; const T* b = (const T*)&remote;
#pragma unroll
      for (int k = 0; k < FragElts; k++) ((T*)&accum[u])[k] = a[k] + b[k];
    }
    for (int dist = 2; dist < nRanks; dist += UnrollPeers) {
      uint4 loaded[UnrollPeers][BULK_UNROLL];
#pragma unroll
      for (int p = 0; p < UnrollPeers; p++) {
        if (dist + p < nRanks) {
          int r = rank + dist + p;
          while (r >= nRanks) r -= nRanks;
          char* peerBase = sendBase + (size_t)r * sendPeerStride + chunkOffset;
#pragma unroll
          for (int u = 0; u < BULK_UNROLL; u++)
            loaded[p][u] = pcclxLsaBulkWarpPeerLoad<PCCLX_LSA_BULK_RMT_C, PCCLX_LSA_BULK_RTMD_PA>(peerBase + u * PCCLX_LSA_BULK_WARP_BYTES);
        }
      }
#pragma unroll
      for (int p = 0; p < UnrollPeers; p++) {
        if (dist + p < nRanks) {
#pragma unroll
          for (int u = 0; u < BULK_UNROLL; u++) {
            T* a = (T*)&accum[u]; const T* b = (const T*)&loaded[p][u];
#pragma unroll
            for (int k = 0; k < FragElts; k++) a[k] += b[k];
          }
        }
      }
    }
#pragma unroll
    for (int u = 0; u < BULK_UNROLL; u++) local[u] = accum[u];
    for (int dist = 0; dist < nRanks; dist += UnrollPeers) {
#pragma unroll
      for (int p = 0; p < UnrollPeers; p++) {
        if (dist + p < nRanks) {
          int r = rank + dist + p;
          while (r >= nRanks) r -= nRanks;
          char* peerBase = recvBase + (size_t)r * recvPeerStride + chunkOffset;
#pragma unroll
          for (int u = 0; u < BULK_UNROLL; u++) {
            if (r == rank) pcclxLsaBulkWarpStore<PCCLX_LSA_BULK_ST_CG>(localRecv + chunkOffset + u * PCCLX_LSA_BULK_WARP_BYTES, local[u]);
            else pcclxLsaBulkWarpPeerStore<PCCLX_LSA_BULK_RMT_C, PCCLX_LSA_BULK_RTMD_PA>(peerBase + u * PCCLX_LSA_BULK_WARP_BYTES, local[u]);
          }
        }
      }
    }
    chunkOffset += (size_t)gwn * BULK_CHUNK_BYTES;
    iters -= gwn;
    if (iters > 0) {
#pragma unroll
      for (int u = 0; u < BULK_UNROLL; u++)
        local[u] = pcclxLsaBulkWarpLoad<PCCLX_LSA_BULK_LD_CG>(localSend + chunkOffset + u * PCCLX_LSA_BULK_WARP_BYTES);
    }
  }
  bar.arrive(pcclCoopCta(), hggc::memory_order_release);
  bar.wait(pcclCoopCta(), hggc::memory_order_relaxed);
}
#endif

testResult_t AllReduceRunCollDevice(void* sendwin, size_t sendoffset, void* recvwin, size_t recvoffset,
    size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclDevComm* devComm, hggcStream_t stream, int deviceImpl) {
  switch(deviceImpl) {
    case 1:
      TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_REDUCE(allReduceLsaKernel, type, op),
                 sendwin, sendoffset, recvwin, recvoffset, count, type, op, root, devComm, stream));
      return testSuccess;
    case 2:
#ifdef PCCLX_HAS_LSA_BULK
      if (devComm->nRanks > 1 && wordSize(type) == 4 && count * wordSize(type) % BULK_CHUNK_BYTES == 0 &&
          sendoffset % 128 == 0 && recvoffset % 128 == 0) {
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_REDUCE(allReduceLsaBulkKernel, type, op),
                   sendwin, sendoffset, recvwin, recvoffset, count, type, op, root, devComm, stream));
        return testSuccess;
      }
#endif
      TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_REDUCE(allReduceLsaVectorizedKernel, type, op),
                 sendwin, sendoffset, recvwin, recvoffset, count, type, op, root, devComm, stream));
      return testSuccess;
    case 3:
    case 4:
      fprintf(stderr, "ERROR: all_reduce device implementation 3/4 (multimem kernels) "
                      "are not supported on PPU.\n");
      return testNotImplemented;
    default:
      fprintf(stderr, "ERROR: all_reduce device implementation %d is not supported.\n", deviceImpl);
      return testNotImplemented;
  }
}
#endif

testResult_t AllReduceRunColl(void* sendbuff, void* recvbuff, size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclComm_t comm, hggcStream_t stream) {
  PCCLCHECK(pcclAllReduce(sendbuff, recvbuff, count, type, op, comm, stream));
  return testSuccess;
}

struct testColl allReduceTest = {
  "AllReduce",
  AllReduceGetCollByteCount,
  AllReduceInitData,
  AllReduceGetBw,
  AllReduceRunColl
};

void AllReduceGetBuffSize(size_t *sendcount, size_t *recvcount, size_t *kerNum, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllReduceGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, 1, nranks);
}

testResult_t AllReduceRunTest(struct threadArgs* args, int root, pcclDataType_t type, const char* typeName, pcclRedOp_t op, const char* opName) {
  args->collTest = &allReduceTest;
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

struct testEngine allReduceEngine = {
  AllReduceGetBuffSize,
  AllReduceRunTest,
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
  AllReduceGetDevCommRequirements,
  AllReduceRunCollDevice
#endif
};

#pragma weak pcclTestEngine=allReduceEngine
