/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include "pccl.h"
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
#include "pccl_device.h"
#endif
#include <stdio.h>
#include <algorithm>
#include "hggc_library_types.h"
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include <pthread.h>

#define HGGCCHECK(cmd) do {                         \
  hggcError_t err = cmd;                            \
  if( err != hggcSuccess ) {                        \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test HGGC failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,hggcGetErrorString(err)); \
    return testHggcError;                           \
  }                                                 \
} while(0)

#define PCCLCHECK(cmd) do {                         \
  pcclResult_t res = cmd;                           \
  if (res != pcclSuccess && res != pcclInProgress) {\
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test PCCL failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,pcclGetErrorString(res)); \
    return testPcclError;                           \
  }                                                 \
} while(0)

// need >= 256 to be different with ascii.
typedef enum {
  OPT_ELAPSED_TYPE = 256,
  OPT_API_TIME,
  OPT_INIT_BLOCKING,
  OPT_RANDOM_SIZE_NUM,
  OPT_SIZE_LIST,
  OPT_INIT_ONLY,
  OPT_CORE_DUMP,
  OPT_ACU_PROFILE,
  OPT_TARGET_PROCESS,
  OPT_ABNORMAL_MODE,
  OPT_BUFFER_OFFSET,
  OPT_REPORT_TYPE,
  OPT_REPORT_CHECK_FILENAME
} pcclTestOpt_t;

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testHggcError = 2,
  testPcclError = 3,
  testHgptiError = 4,
  testNotImplemented = 5,
  testInvalidUsage = 6
} testResult_t;

typedef enum {
  REP_DEFAULT = 0,
  REP_PRECISE = 1,
  REP_DETAIL = 2
} reportType_t;

// Relay errors up and trace
#define TESTCHECK(cmd) do {                         \
  testResult_t r = cmd;                             \
  if (r!= testSuccess) {                            \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf(" .. %s: Test failure %s:%d\n",          \
         hostname,                                  \
        __FILE__,__LINE__);                         \
    return r;                                       \
  }                                                 \
} while(0)

struct testColl {
  const char name[20];
  void (*getCollByteCount)(
      size_t *sendcount, size_t *recvcount, size_t *paramcount,
      size_t *sendInplaceOffset, size_t *recvInplaceOffset,
      size_t count, size_t eltSize, int nranks);
  testResult_t (*initData)(struct threadArgs* args, pcclDataType_t type,
      pcclRedOp_t op, int root, int rep, int in_place);
  void (*getBw)(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);
  testResult_t (*runColl)(void* sendbuff, void* recvbuff, size_t count, pcclDataType_t type,
      pcclRedOp_t op, int root, pcclComm_t comm, hggcStream_t stream);
};
extern struct testColl allReduceTest;
extern struct testColl allGatherTest;
extern struct testColl reduceScatterTest;
extern struct testColl broadcastTest;
extern struct testColl reduceTest;

struct testEngine {
  void (*getBuffSize)(size_t *sendcount, size_t *recvcount, size_t *kerNum, size_t count, int nranks);
  testResult_t (*runTest)(struct threadArgs* args, int root, pcclDataType_t type,
      const char* typeName, pcclRedOp_t op, const char* opName);
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
  testResult_t (*getDevCommRequirements)(int deviceImpl, pcclDevCommRequirements* reqs, pcclComm_t comm);
  testResult_t (*runCollDevice)(void* sendwin, size_t sendoffset, void* recvwin, size_t recvoffset,
      size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclDevComm* devComm, hggcStream_t stream, int deviceImpl);
#endif
};

extern struct testEngine pcclTestEngine;

struct threadArgs {
  size_t nbytes;
  size_t minbytes;
  size_t maxbytes;
  size_t stepbytes;
  size_t stepfactor;
  size_t *sizelist;

  int totalProcs;
  int nProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  int* gpus;
  int localRank;
  void** sendbuffs;
  size_t sendBytes;
  size_t sendInplaceOffset;
  void** recvbuffs;
  size_t recvInplaceOffset;
  pcclUniqueId* pcclIds;
  pcclComm_t* comms;
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
  pcclDevComm* devComms;
#endif
  hggcStream_t* streams;
  hggcEvent_t* startEvents;
  hggcEvent_t* stopEvents;

  void** sendRegHandles;
  void** recvRegHandles;

  void** expected;
  size_t expectedBytes;
  volatile int* barrier;
  int barrier_idx;
  int syncRank;
  int syncNranks;
  int* errors;
  double* bw;
  int* bw_count;

  int reportErrors;

  struct testColl* collTest;
  int blocking;
};

typedef testResult_t (*threadFunc_t)(struct threadArgs* args);
struct testThread {
  pthread_t thread;
  threadFunc_t func;
  struct threadArgs args;
  testResult_t ret;
};

#include <chrono>

// Provided by common.cu
extern void Barrier(struct threadArgs* args);
extern testResult_t TimeTest(struct threadArgs* args, pcclDataType_t type, const char* typeName, pcclRedOp_t op,  const char* opName, int root);
extern testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, pcclDataType_t type, pcclRedOp_t op, const uint64_t seed, const int nranks);
extern testResult_t InitData(void* data, const size_t count, size_t offset, pcclDataType_t type, pcclRedOp_t op, const uint64_t seed, const int nranks, const int rank);
extern void AllocateBuffs(void **sendbuff, void **recvbuff, void **expected, void **expectedHost, size_t nbytes, int nranks);
extern testResult_t waitReady(struct threadArgs* args);

// Provided by each coll
extern void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root);
extern void print_header(reportType_t rep_type);

#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

#include <stdint.h>

static uint64_t getHash(const char* string, size_t n) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (size_t c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash(const char* hostname) {
  char hostHash[1024];

  // Fall back is the hostname if something fails
  (void) strncpy(hostHash, hostname, sizeof(hostHash));
  int offset = strlen(hostHash);

  FILE *file = fopen(HOSTID_FILE, "r");
  if (file != NULL) {
    char *p;
    if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        free(p);
    }
  }
  fclose(file);

  // Make sure the string is terminated
  hostHash[sizeof(hostHash)-1]='\0';

  return getHash(hostHash, strlen(hostHash));
}

#define HAVE_BF16 0
#define HAVE_FP8 0

#if defined(__HGGC_BF16_TYPES_EXIST__)
  #undef HAVE_BF16
  #define HAVE_BF16 1
#endif
#if defined(__HGGC_FP8_TYPES_EXIST__) && PCCL_VERSION_CODE >= PCCL_VERSION(1,7,0)
  #undef HAVE_FP8
  #define HAVE_FP8 1
#endif

static size_t wordSize(pcclDataType_t type) {
  switch(type) {
    case pcclInt8:
    case pcclUint8:
#if HAVE_FP8
    case pcclFloat8e4m3:
    case pcclFloat8e5m2:
#endif
      return 1;
#if HAVE_BF16
    case pcclBfloat16:
#endif
    case pcclFloat16:
      return 2;
    case pcclInt32:
    case pcclUint32:
    case pcclFloat32:
      return 4;
    case pcclInt64:
    case pcclUint64:
    case pcclFloat64:
      return 8;
    default: return 0;
  }
}

extern int test_pcclVersion; // init'd with pcclGetVersion()
extern int test_opnum;
extern int test_typenum;
extern pcclDataType_t test_types[pcclNumTypes];
extern const char *test_typenames[pcclNumTypes];
extern pcclRedOp_t test_ops[];
extern const char *test_opnames[];

static int pcclstringtotype(char *str) {
    for (int t=0; t<pcclNumTypes; t++) {
      if (strcmp(str, test_typenames[t]) == 0) {
        return t;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid type %s, defaulting to %s .. \n", str, test_typenames[pcclFloat32]);
    return pcclFloat32;
}

static int pcclstringtoop (char *str) {
    for (int o=0; o<test_opnum; o++) {
      if (strcmp(str, test_opnames[o]) == 0) {
        return o;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid op %s, defaulting to %s .. \n", str, test_opnames[pcclSum]);
    return pcclSum;
}

static int getSeed(const char* collName, int rep, int rank) {
  int seed = rep;
  if (!strcmp(collName, "AllGather") || !strcmp(collName, "AlltoAll") || !strcmp(collName, "HyperCube")) {
    seed = 33*rep + rank;
  }
  return seed;
}

extern thread_local int is_main_thread;
#define PRINT if (is_main_thread) printf

extern int deviceCtaCount;

#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
template <typename F>
testResult_t testLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, pcclDataType_t type, pcclRedOp_t op, int root, pcclDevComm* devComm, hggcStream_t stream) {
  if (kernel == nullptr) return testNotImplemented;
  pcclWindow_t sendwin = (pcclWindow_t)sendbuff;
  pcclWindow_t recvwin = (pcclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm);
  return testSuccess;
}

#if defined(__HGGC_BF16_TYPES_EXIST__)
#define SPECIALIZE_KERNEL_BF16(kernel) \
   type == pcclBfloat16 ? kernel<__ppu_bfloat16> :
#else
#define SPECIALIZE_KERNEL_BF16(kernel)
#endif

#if defined(__HGGC_FP8_TYPES_EXIST__)
#define SPECIALIZE_KERNEL_FP8(kernel) \
   type == pcclFloat8e4m3 ? kernel<__hg_fp8_e4m3> : \
   type == pcclFloat8e5m2 ? kernel<__hg_fp8_e5m2> :
#else
#define SPECIALIZE_KERNEL_FP8(kernel)
#endif

#define SPECIALIZE_KERNEL(kernel, type, op) \
  ( op != pcclSum ? nullptr : \
   type == pcclInt8 ? kernel<int8_t> : \
   type == pcclUint8 ? kernel<uint8_t> : \
   type == pcclInt32 ? kernel<int32_t> : \
   type == pcclUint32 ? kernel<uint32_t> : \
   type == pcclInt64 ? kernel<int64_t> : \
   type == pcclUint64 ? kernel<uint64_t> : \
   type == pcclFloat16 ? kernel<half> : \
   type == pcclFloat32 ? kernel<float> : \
   type == pcclFloat64 ? kernel<double> : \
   SPECIALIZE_KERNEL_BF16(kernel) \
   SPECIALIZE_KERNEL_FP8(kernel) \
   nullptr \
  )

#define SPECIALIZE_KERNEL_REDUCE(kernel, type, op) \
  ( op != pcclSum ? nullptr : \
   type == pcclInt8 ? kernel<int8_t> : \
   type == pcclUint8 ? kernel<uint8_t> : \
   type == pcclInt32 ? kernel<int32_t> : \
   type == pcclUint32 ? kernel<uint32_t> : \
   type == pcclInt64 ? kernel<int64_t> : \
   type == pcclUint64 ? kernel<uint64_t> : \
   type == pcclFloat16 ? kernel<half> : \
   type == pcclFloat32 ? kernel<float> : \
   type == pcclFloat64 ? kernel<double> : \
   SPECIALIZE_KERNEL_BF16(kernel) \
   nullptr \
  )

#endif

#endif
