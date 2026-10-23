/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <pthread.h>
#include <sched.h>
#include <cstdio>
#include <getopt.h>
#include <libgen.h>
#include "hggc.h"
#include <sys/resource.h>
#include <hggc_profiler_api.h> // for Asight Compute profiling
#include <stdlib.h>
#include <limits.h>
#include "common.h"
#ifndef DISABLE_HGPTI
#include "hgpti_common.h"
#endif
#include <random>

#include "../verifiable/verifiable.h"

#pragma weak pcclCommWindowRegister
#pragma weak pcclCommWindowDeregister
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
#pragma weak pcclDevCommCreate
#pragma weak pcclDevCommDestroy
#pragma weak pcclCommQueryProperties
#endif

#ifndef MPI_SUPPORT
#include "barrier.h"
#endif

int test_pcclVersion = 0; // init'd with pcclGetVersion()

// Number of base data types (excluding BF16 and FP8)
#define PCCL_NUM_BASE_TYPES 9
pcclDataType_t test_types[pcclNumTypes] = {pcclInt8, pcclUint8, pcclInt32, pcclUint32, pcclInt64, pcclUint64, pcclFloat16, pcclFloat32, pcclFloat64,
#if HAVE_BF16
  pcclBfloat16,
#endif
#if HAVE_FP8
  pcclFloat8e4m3,
  pcclFloat8e5m2
#endif
};
const char *test_typenames[pcclNumTypes] = {"int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double",
#if HAVE_BF16
  "bf16",
#endif
#if HAVE_FP8
  "f8e4m3",
  "f8e5m2"
#endif
};

int test_typenum = -1;
int test_opnum = pcclNumOps+1;
pcclRedOp_t test_ops[] = {pcclSum, pcclProd, pcclMax, pcclMin, pcclAvg,
                          pcclNumOps // stand in for pcclRedOpCreatePreMulSum() created on-demand
                          };
const char *test_opnames[] = {"sum", "prod", "max", "min", "avg", "mulsum"};
const char *averageStr[4] = {"RANK0", "AVG", "MIN", "MAX"};

enum { INIT_CONFIG=0, INIT_PARALLEL=1, INIT_SCALABLE=2 };

thread_local int is_main_thread = 0;

#define MAX_SIZE_LIST 2048

// Command line parameter defaults
static int nThreads = 1;
static int nGpus = 1;
static int size_list_num = 0;
static size_t minBytes = 32*1024*1024;
static size_t maxBytes = 32*1024*1024;
static size_t stepBytes = 1*1024*1024;
static size_t stepFactor = 1;
static size_t randomSizeNum = 0;
static size_t listBytes[MAX_SIZE_LIST];
static size_t sendbuff_offset = 0;
static size_t recvbuff_offset = 0;
static int datacheck = 1;
static int single_test = 0;
// 0 means inplace buffer used to benchmark, 1 means outplace buffer used to benchmark,
// 2 means both of inplace and outplace buffers used to benchmark
static int test_buffer_kind = 2;
static int warmup_iters = 5;
static int iters = 20;
static int abnormal_mode = 0;
static int agg_iters = 1;
static int pcclop = pcclSum;
static int pccltype = pcclFloat32;
static int pcclroot = 0;
static int init_type = INIT_CONFIG;
static int blocking_coll = 0;
static int hggcGraphLaunches = 0;
static int core_dump = 0;
// 0 means elapsed cpu time, 1 means elapsed hggc event time 2 means hgpti time
static int elapsed_type = 0;
// 1 means using Asight Compute profiling
static int acu_profile = 0;
static reportType_t rep_type = REP_DEFAULT;
static int init_only = 0;
static int init_blocking = 1;
static int init_blocking_timeout = 600; // default 10m
static int target_proc = -1;
#define LOCAL_REGISTER 1
#define SYMMETRIC_REGISTER 2
static int local_register = 0;
static int ctaPolicy = -1;
static int minHggcArch = 1<<30;
// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
static int average = 2;
static char* report_check_filename = NULL;
static bool api_time = false;
static int deviceImpl = 0;
int deviceCtaCount = 16;
static bool deviceCtaCountSet = false;
static int deviceMpCount = 0;

inline void print_header(reportType_t rep_type) {
  const char* timeStr = (elapsed_type == 0) ? " cpu_time " : (elapsed_type == 1) ? "event_time" : "hgpti_time";
  if (rep_type > REP_DEFAULT) {
    PRINT("#\n");
    PRINT("# %10s  %12s  %8s  %6s  %5s"
          "                                      out-of-place"
          "                                                               in-place\n",
          "", "", "", "", "");
    PRINT("# %10s  %12s  %8s  %6s  %5s"
          "  %7s  %5s  %6s  %5s  %6s  %6s  %6s  %6s  %7s"
          "  %7s  %5s  %6s  %5s  %6s  %6s  %6s  %6s  %7s\n",
          "size", "count", "type", "redop", "root",
          timeStr, "algbw", "busbw", "min", "p90", "p99", "max", "var", "#wrong",
          timeStr, "algbw", "busbw", "min", "p90", "p99", "max", "var", "#wrong");
    PRINT("# %10s  %12s  %8s  %6s  %5s"
          "  %7s  %8s  %6s  %6s  %6s  %6s  %6s  %6s  %7s"
          "  %6s  %8s  %6s  %6s  %6s  %6s  %6s  %6s  %7s\n",
          "(B)", "(elements)", "", "", "",
          "(us)", "(GB/s)", "(GB/s)", "", "", "", "", "", "",
          "(us)", "(GB/s)", "(GB/s)", "", "", "", "", "", "");
  } else {
    PRINT("#\n");
    PRINT("# %9s  %12s  %8s  %6s  %5s"
          "             out-of-place"
          "                      in-place\n",
          "", "", "", "", "");

    const char* hostTimeStr = (api_time) ? "api_time" : "";
    const char* hostTimeUnitStr = (api_time) ? "    (us)" : "";
    PRINT("# %10s  %12s  %8s  %6s  %5s"
          "  %7s %s %5s  %5s  %7s"
          "  %7s %s %5s  %5s  %7s\n",
          "size", "count", "type", "redop", "root",
          timeStr, hostTimeStr, "algbw", "busbw", "#wrong",
          timeStr, hostTimeStr, "algbw", "busbw", "#wrong");
    PRINT("# %10s  %12s  %8s  %6s  %5s"
          "  %7s %s %8s  %6s  %7s"
          "  %6s %s %8s  %6s  %7s\n",
          "(B)", "(elements)", "", "", "",
          "(us)", hostTimeUnitStr, "(GB/s)", "(GB/s)", "",
          "(us)", hostTimeUnitStr, "(GB/s)", "(GB/s)", "");
  }
}

__global__ void dummy_kernel() {
  for (int i = 0; i < 100; i++) __nanosleep(1000000);  // 100ms
}

static inline double elapsedTime(std::chrono::time_point<std::chrono::high_resolution_clock> start, int nIters) {
  double deltaSec = 0.0;
  auto delta = std::chrono::high_resolution_clock::now() - start;
  deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
  deltaSec = deltaSec / nIters;
  return deltaSec;
}

double parsesize(char *value) {
    long long int units;
    double size;

    if (strchr(value, 'G') != NULL) {
        units=1024*1024*1024;
    } else if (strchr(value, 'M') != NULL) {
        units=1024*1024;
    } else if (strchr(value, 'K') != NULL) {
        units=1024;
    } else {
        units=1;
    }

    size = atof(value)*units;
    return size;
}

void parsesizelist(char *param) {
  size_t minListBytes = ULONG_MAX;
  size_t maxListBytes = 0;
  char *vals = strtok(param, ",");
  int idx = 0;
  while (vals != NULL) {
    size_t val = (size_t)parsesize(vals);
    listBytes[idx++] = val;
    minListBytes = val < minListBytes ? val : minListBytes;
    maxListBytes = val > maxListBytes ? val : maxListBytes;
    vals = strtok(NULL, ",");
  }
  if (idx > 0) {
    size_list_num = idx;
    minBytes = minListBytes;
    maxBytes = maxListBytes;
  }
  for (;idx < MAX_SIZE_LIST; idx++) listBytes[idx] = 0;
}

bool rankHasRoot(const int rank, const int nRanks, const int nIds) {
  const int rmr = nRanks % nIds;
  const int rpr = nRanks / nIds;
  const int rlim = rmr * (rpr+1);
  if (rank < rlim) {
    return !(rank % (rpr + 1));
  } else {
    return !((rank - rlim) % rpr);
  }
}

// Query InProgress done
testResult_t waitReady(struct threadArgs* args) {
  if (args->blocking) return  testSuccess;

  int commCnts = args->nGpus;
  testResult_t ret = testSuccess;
  pcclResult_t state = pcclSuccess;
  auto startTime = std::chrono::steady_clock::now();
  while (commCnts) {
    PCCLCHECK(pcclCommGetAsyncError(args->comms[commCnts-1], &state));
    if (state == pcclInProgress) {
      if (init_blocking_timeout) {
        auto curTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime).count();
        if (elapsedTime > init_blocking_timeout) {
          ret = testInternalError;
          PRINT("waitReady timeout !");
          break;
        }
      }
      continue;
    }
    // already done
    --commCnts;
  }

  if (ret == testInternalError) {
    for (int i=0; i<args->nGpus; i++)
      PCCLCHECK(pcclCommAbort(args->comms[i]));
  }
  return ret;
}

testResult_t CheckDelta(void* results, void* expected, size_t count, size_t offset, pcclDataType_t type, pcclRedOp_t op, uint64_t seed, int nranks, int64_t *wrongEltN) {
  HGGCCHECK(pcclVerifiableVerify(results, expected, count, (int)type, (int)op, nranks, seed, offset, wrongEltN, hggcStreamDefault));
  HGGCCHECK(hggcDeviceSynchronize());
  return testSuccess;
}

testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, pcclDataType_t type, pcclRedOp_t op, uint64_t seed, int nranks) {
  HGGCCHECK(pcclVerifiablePrepareExpected(data, count, (int)type, (int)op, nranks, seed, offset, hggcStreamDefault));
  return testSuccess;
}

testResult_t InitData(void* data, const size_t count, size_t offset, pcclDataType_t type, pcclRedOp_t op, uint64_t seed, int nranks, int rank) {
  HGGCCHECK(pcclVerifiablePrepareInput(data, count, (int)type, (int)op, nranks, rank, seed, offset, hggcStreamDefault));
  return testSuccess;
}

void Barrier(struct threadArgs* args) {
  while (args->barrier[args->barrier_idx] != args->thread) sched_yield();

  args->barrier[args->barrier_idx] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
    if (args->nProcs > 1) {
#ifdef MPI_SUPPORT
    MPI_Barrier(MPI_COMM_WORLD);
#else
    barrier(args->proc, args->nProcs);
#endif
    }
    args->barrier[args->barrier_idx] = 0;
  } else {
    while (args->barrier[args->barrier_idx]) sched_yield();
  }

  args->barrier_idx=!args->barrier_idx;
}

const double EPSILON = 1e-2;
bool isCloseToZero(double value, double epsilon = EPSILON) {
    return abs(value) < epsilon;
}

// for average=0, '-x 1' will return event time on rank 0 but '-x 0' will only return cpu time on thread 0,
// they are not equivalent when nGpus != 1
void Allreduce(struct threadArgs *args, double *value, double *valuePerGpus, int valueCount, int average, int nThreads) {
  static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
  static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  static double accumulator;
  static int counter;

  accumulator = 0.0;
  counter = 0;
  if (nThreads != 1) Barrier(args);
#ifdef MPI_SUPPORT
  else if (args->nProcs > 1) MPI_Barrier(MPI_COMM_WORLD);
#endif

  pthread_mutex_lock(&lock);
  for (int i = 0; i < valueCount; i++) {
    switch(average) {
      case /*r0*/ 0: if(args->thread == 0 && i == 0) accumulator = valuePerGpus[i]; break;
      case /*avg*/1: accumulator += valuePerGpus[i]; break;
      case /*min*/2: accumulator = isCloseToZero(accumulator) ? valuePerGpus[i] :
                                          std::min<double>(accumulator, valuePerGpus[i]); break;
      case /*max*/3: accumulator = std::max<double>(accumulator, valuePerGpus[i]); break;
      case /*sum*/4: accumulator += valuePerGpus[i]; break;
    }
  }

  if(++counter == nThreads)
    pthread_cond_broadcast(&cond);

  if(args->thread+1 == nThreads) {
    while(counter != nThreads)
      pthread_cond_wait(&cond, &lock);

#ifdef MPI_SUPPORT
    if(average != 0) {
      MPI_Op op = average == 1 ? MPI_SUM :
                  average == 2 ? MPI_MIN :
                  average == 3 ? MPI_MAX :
                  average == 4 ? MPI_SUM : MPI_Op();
      MPI_Allreduce(MPI_IN_PLACE, (void*)&accumulator, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
    }
#else
    if (args->nProcs > 1 && average != 0) {
      allreduceFromSocket(&accumulator, args->proc, average);
    }
#endif

    if(average == 1) accumulator /= args->totalProcs*nThreads*valueCount;
    counter = 0;
    pthread_cond_broadcast(&cond);
  } else {
    while(counter != 0)
      pthread_cond_wait(&cond, &lock);
  }
  pthread_mutex_unlock(&lock);

  *value = accumulator;
}

static inline void getSourceRankSourceIndex(struct testColl* collTest, size_t expectedBytes, int root, int rank, int nRanks, size_t byteIndex, int* sourceRank, size_t* sourceByteIndex) {
  const char* collName = collTest->name;
  *sourceRank = 0;
  *sourceByteIndex = 0;
  size_t bytesPerRank = expectedBytes/nRanks;
  if (collName == NULL) return;
  if (!strcmp(collName, "AllGather") || !strcmp(collName, "AlltoAll") || !strcmp(collName, "Gather") || !strcmp(collName, "HyperCube")) {
    int r = 0;
    while (byteIndex >= (r+1)*bytesPerRank) {r++;}
    *sourceRank = r;
    if (!strcmp(collName, "AllGather") || !strcmp(collName, "HyperCube")) {
      *sourceByteIndex = byteIndex - r*bytesPerRank;
    } else if (!strcmp(collName, "AlltoAll")) {
      *sourceByteIndex = byteIndex + (rank-r)*bytesPerRank;
    } else { // Gather
      *sourceByteIndex = byteIndex;
    }
  } else if (!strcmp(collName, "ReduceScatter") || !strcmp(collName, "Scatter")) {
    *sourceRank = root; // will not be used
    *sourceByteIndex = byteIndex + rank*bytesPerRank;
  } else if (!strcmp(collName, "SendRecv")) {
    *sourceRank = (rank-1+nRanks)%nRanks;
    *sourceByteIndex = byteIndex + (*sourceRank)*expectedBytes;
  } else { // AllReduce, Broadcast, Reduce
    *sourceRank = root;
    *sourceByteIndex = byteIndex;
  }
  return;
}

static inline uint32_t loadU32Le(const unsigned char* p) {
  return ((uint32_t)p[0]) |
         ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static inline void writeDataToFile(FILE* fp, size_t size, unsigned char* data, bool isFloat, float* outFloat = NULL) {
  uint32_t bits;
  float f;
  if (isFloat) {
    bits = loadU32Le(data);
    memcpy(&f, &bits, sizeof(f));
    if (outFloat) *outFloat = f;
  }
  fprintf(fp, "0x");
  for (int k=0; k<size; k++) {
    fprintf(fp, "%02x", data[k]);
  }
  if (isFloat)
    fprintf(fp, " (%.9e) ", f);
  return;
}

static inline void printSourceData3ColsForFloat(FILE* fp, int rank, int nranks, pcclDataType_t type, pcclRedOp_t op, int seed, size_t sourceIndex) {
  uint32_t bits;
  float f;
  fprintf(fp, "SourceData (float32):\n");
  fprintf(fp, "%-6s %-12s %s\n", "rank", "bits", "value");
  fprintf(fp, "%-6s %-12s %s\n", "----", "------------", "----------------------");
  for (int i = 0; i < nranks; i++) {
    alignas(8) unsigned char buf[4];
    pcclVerifiableGetInput(buf, type, op, nranks, i, seed, sourceIndex);
    bits = loadU32Le(buf);
    memcpy(&f, &bits, sizeof(f));
    fprintf(fp, "%-6d 0x%02x%02x%02x%02x  % .9e\n", i, buf[0], buf[1], buf[2], buf[3], f);
  }
  return;
}


static inline void writeDataCheckReport(FILE* fp, struct testColl* collTest, pcclDataType_t type, pcclRedOp_t op, int rank, int nranks, unsigned char* expectedHost, unsigned char* dataHost, size_t index, int sourceRank, size_t sourceIndex, int rep) {
  bool isFloat = type == pcclFloat32;
  const char* collName = collTest->name;
  int seed = getSeed(collName, rep, sourceRank);
  if (collName == NULL) return;
  bool isReduce = (!strcmp(collName, "Reduce") || !strcmp(collName, "ReduceScatter") || !strcmp(collName, "AllReduce"));
  float expect, actual;
  fprintf(fp, "%s rank %d, index %lu: expect ", collName, rank, index);
  writeDataToFile(fp, wordSize(type), expectedHost, isFloat, &expect);
  fprintf(fp, " actual ");
  writeDataToFile(fp, wordSize(type), dataHost, isFloat, &actual);
  if (isFloat) fprintf(fp, " (expect - actual) diff is %.9e", (expect - actual));
  fprintf(fp, "\n");
  if (isReduce) {
    if (isFloat) {
      printSourceData3ColsForFloat(fp, rank, nranks, type, op, seed, sourceIndex);
    } else {
      fprintf(fp, "Here are source data: ");
      for (int i = 0; i < nranks; i++) {
        alignas(8) unsigned char buf[8];
        fprintf(fp, "rank %d ", i);
        pcclVerifiableGetInput(buf, type, op, nranks, i, seed, sourceIndex);
        writeDataToFile(fp, wordSize(type), buf, isFloat);
      }
    }
  } else {
    alignas(8) unsigned char buf[8];
    fprintf(fp, "data should come from source rank %d, ", sourceRank);
    pcclVerifiableGetInput(buf, type, op, 1, 0, seed, sourceIndex);
    fprintf(fp, "value is ");
    writeDataToFile(fp, wordSize(type), buf, isFloat);
  }
  fprintf(fp, "\n");
  fprintf(fp, "\n");
  return;
}

testResult_t CheckData(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t op, int root, int in_place, double *nWrongEltPerThread, int rep) {
  int nranks = args->nProcs*args->nGpus*args->nThreads;
  size_t count = args->expectedBytes/wordSize(type);

  int64_t *nWrongEltPerGpu = NULL;
  HGGCCHECK(hggcHostAlloc((void**)&nWrongEltPerGpu, args->nGpus*sizeof(int64_t), hggcHostAllocMapped));

  for (int i=0; i<args->nGpus; i++) {
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    void *data = in_place ? ((void *)((uintptr_t)args->recvbuffs[i] + args->recvInplaceOffset*rank)) : args->recvbuffs[i];

    nWrongEltPerGpu[i] = 0;
    TESTCHECK(CheckDelta(data, args->expected[i], count, 0, type, op, 0, nranks, nWrongEltPerGpu+i));

    if (args->reportErrors && (report_check_filename != NULL && nWrongEltPerGpu[i] > 0)) {
      char report_filename[256];
      snprintf(report_filename, sizeof(report_filename), "%s_%d.log", report_check_filename, rank);
      FILE *fp = fopen(report_filename, "a");
      if (fp != NULL) {
        unsigned char *expectedHost = (unsigned char *)malloc(args->expectedBytes);
        unsigned char *dataHost = (unsigned char *)malloc(args->expectedBytes);
        HGGCCHECK(hggcMemcpy(expectedHost, args->expected[i], args->expectedBytes, hggcMemcpyDeviceToHost));
        HGGCCHECK(hggcMemcpy(dataHost, data, args->expectedBytes, hggcMemcpyDeviceToHost));
        fprintf(fp, "------ start check, rank %d, datatype %d, op %d, recvbuff size %lu, isInPlace %d ------\n", rank, type, (int)op, args->expectedBytes, in_place);
        size_t wrong_count = 0;
        for (size_t j=0; j<args->expectedBytes; j+=wordSize(type)) {
          bool checkPass = true;
          for (int k=0; k<wordSize(type); k++) {
            if (expectedHost[j+k] != dataHost[j+k]) {checkPass = false; break;}
          }
          if (!checkPass) {
            int sourceRank = -1;
            size_t sourceByteIndex = 0;
            getSourceRankSourceIndex(args->collTest, args->expectedBytes, root, rank, nranks, j, &sourceRank, &sourceByteIndex);
            writeDataCheckReport(fp, args->collTest, type, op, rank, nranks, expectedHost+j, dataHost+j, j/wordSize(type), sourceRank, sourceByteIndex/wordSize(type), rep);
            wrong_count++;
          }
        }
        fprintf(fp, "------ end check recvbuff size %lu, kernel check wrong_count %lu, host check wrong_count %lu ------\n\n", args->expectedBytes, nWrongEltPerGpu[i], wrong_count);
        free(expectedHost);
        free(dataHost);
      }
      fclose(fp);
    }
  }
  *nWrongEltPerThread = 0;
  for (int i=0; i<args->nGpus; i++) *nWrongEltPerThread += nWrongEltPerGpu[i];
  hggcFreeHost(nWrongEltPerGpu);
  if (args->reportErrors && *nWrongEltPerThread) args->errors[0]++;
  return testSuccess;
}

testResult_t testStreamSynchronize(struct threadArgs* args) {
  int ngpus = args->nGpus;
  hggcStream_t* streams = args->streams;
  pcclComm_t* comms = args->comms;
  hggcError_t hggcErr;
  int remaining = ngpus;
  int* done = (int*)malloc(sizeof(int)*ngpus);

  TESTCHECK(waitReady(args));
  memset(done, 0, sizeof(int)*ngpus);

  while (remaining) {
    int idle = 1;
    for (int i=0; i<ngpus; i++) {
      if (done[i]) continue;

      hggcErr = hggcStreamQuery(streams[i]);
      if (hggcErr == hggcSuccess) {
        done[i] = 1;
        remaining--;
        idle = 0;
        continue;
      }

      if (hggcErr != hggcErrorNotReady) HGGCCHECK(hggcErr);

      if (comms) {
        pcclResult_t pcclAsyncErr;
        PCCLCHECK(pcclCommGetAsyncError(comms[i], &pcclAsyncErr));
        if (pcclAsyncErr != pcclSuccess) {
          // An asynchronous error happened. Stop the operation and destroy
          // the communicator
          for (int i=0; i<ngpus; i++)
            PCCLCHECK(pcclCommAbort(comms[i]));
          // Abort the perf test
          PCCLCHECK(pcclAsyncErr);
        }
      }
    }

    // We might want to let other threads (including PCCL threads) use the CPU.
    if (idle) sched_yield();
  }
  free(done);
  return testSuccess;
}

testResult_t startColl(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t opIndex, int root, int in_place, int iter) {
  size_t count = args->nbytes / wordSize(type);

  // Try to change offset for each iteration so that we avoid cache effects and catch race conditions in ptrExchange
  size_t totalnbytes = max(args->sendBytes, args->expectedBytes);
  size_t shift = (totalnbytes * iter) % args->maxbytes;
  shift = shift/wordSize(type)* wordSize(type);  // sendbuff/recvbuff should always align with type;
  if (shift + totalnbytes > args->maxbytes) shift = 0;
  static int launched_iter = 0;

  TESTCHECK(waitReady(args));
  if (args->nGpus > 1) PCCLCHECK(pcclGroupStart());
  for (int i = 0; i < args->nGpus; i++) {
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    char* recvBuff = ((char*)args->recvbuffs[i]) + shift;
    char* sendBuff = ((char*)args->sendbuffs[i]) + shift;
    pcclDataType_t realType = type;
    size_t realCount = count;
    pcclRedOp_t op = opIndex;

    if(opIndex >= pcclNumOps) {
      union {
        int8_t i8; uint8_t u8; int32_t i32; uint32_t u32; int64_t i64; uint64_t u64;
        half f16; float f32; double f64;
#if HAVE_BF16
        __ppu_bfloat16 bf16;
#endif
#if HAVE_FP8
        __hg_fp8_e4m3 f8e4m3;
        __hg_fp8_e5m2 f8e5m2;
#endif
      };
      switch(type) {
        case pcclInt8: i8 = pcclVerifiablePremulScalar<int8_t>(rank); break;
        case pcclUint8: u8 = pcclVerifiablePremulScalar<uint8_t>(rank); break;
        case pcclInt32: i32 = pcclVerifiablePremulScalar<int32_t>(rank); break;
        case pcclUint32: u32 = pcclVerifiablePremulScalar<uint32_t>(rank); break;
        case pcclInt64: i64 = pcclVerifiablePremulScalar<int64_t>(rank); break;
        case pcclUint64: u64 = pcclVerifiablePremulScalar<uint64_t>(rank); break;
        case pcclFloat16: f16 = pcclVerifiablePremulScalar<half>(rank); break;
        case pcclFloat32: f32 = pcclVerifiablePremulScalar<float>(rank); break;
        case pcclFloat64: f64 = pcclVerifiablePremulScalar<double>(rank); break;
#if HAVE_BF16
        case pcclBfloat16: bf16 = pcclVerifiablePremulScalar<__ppu_bfloat16>(rank); break;
#endif
#if HAVE_FP8
        case pcclFloat8e4m3: f8e4m3 = pcclVerifiablePremulScalar<__hg_fp8_e4m3>(rank); break;
        case pcclFloat8e5m2: f8e5m2 = pcclVerifiablePremulScalar<__hg_fp8_e5m2>(rank); break;
#endif
        default: break; // Just to silence clang
      }
      PCCLCHECK(pcclRedOpCreatePreMulSum(&op, &u64, type, pcclScalarHostImmediate, args->comms[i]));
    }

    // for abnormal mode1, after 2 iterations, change every rank's datatype, although
    // all Ranks can still finish prims setup.
    // but at algo level, each rank use different type which would influence the count to trigger a hang.
    if (launched_iter >= 2 ) {
      switch(abnormal_mode) {
        case 1:
          realType = (pcclDataType_t)(args->gpus[i]%pcclNumTypes);
          break;
        case 2:
          if (args->gpus[i] != 0) realCount /= args->gpus[i];
          if (realCount < 0) realCount = count;
          break;
      }
    }
    if (deviceImpl == 0) {
      TESTCHECK(args->collTest->runColl(
            (void*)(in_place ? recvBuff + args->sendInplaceOffset*rank : sendBuff),
            (void*)(in_place ? recvBuff + args->recvInplaceOffset*rank : recvBuff),
          realCount, realType, op, root, args->comms[i], args->streams[i]));
    }
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
    else {
      void* sendwin = args->sendRegHandles ? args->sendRegHandles[i] : NULL;
      void* recvwin = args->recvRegHandles ? args->recvRegHandles[i] : NULL;
      HGGCCHECK(hggcSetDevice(args->gpus[i]));
      TESTCHECK(pcclTestEngine.runCollDevice(
            (void*)(in_place ? recvwin : sendwin),
            shift + (in_place ? args->sendInplaceOffset*rank : 0),
            (void*)recvwin,
            shift + (in_place ? args->recvInplaceOffset*rank : 0),
            realCount, realType, op, root, args->devComms+i, args->streams[i], deviceImpl));
    }
#endif

    if(opIndex >= pcclNumOps) {
      PCCLCHECK(pcclRedOpDestroy(op, args->comms[i]));
    }
  }
  if (abnormal_mode == 3 && args->proc % 2 == 0) sleep(80);
  launched_iter++;
  if (args->nGpus > 1) PCCLCHECK(pcclGroupEnd());

  if (blocking_coll) {
    // Complete op before returning
    TESTCHECK(testStreamSynchronize(args));
  }
  if (blocking_coll) Barrier(args);
  return testSuccess;
}

testResult_t completeColl(struct threadArgs* args) {
  if (blocking_coll) return testSuccess;
  TESTCHECK(testStreamSynchronize(args));
  return testSuccess;
}

testResult_t BenchTime(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t op, int root, int in_place, double *timeUsec, double *apiTimeUsec) {
  Barrier(args);
  if (elapsed_type == 1) {
    int savedDev;
    HGGCCHECK(hggcGetDevice(&savedDev));
    for (int i = 0; i < args->nGpus; i++) {
      HGGCCHECK(hggcSetDevice(args->gpus[i]));
      HGGCCHECK(hggcLaunchKernel((void*)dummy_kernel, 32, 1, NULL, 0, args->streams[i]));
    }
    HGGCCHECK(hggcSetDevice(savedDev));
  }

 /************************** Performance Benchmark **************************/
  // record start chrono clock
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  if (elapsed_type == 1) {
    for (int i = 0; i < nGpus; i++) {
      HGGCCHECK(hggcEventRecord(args->startEvents[i], args->streams[i]));
    }
  }

  start = std::chrono::high_resolution_clock::now();
  hggcGraph_t graphs[args->nGpus];
  hggcGraphExec_t graphExec[args->nGpus];
  if (hggcGraphLaunches >= 1) {
    // Begin hggc graph capture
    for (int i = 0; i < args->nGpus; i++) {
      // Thread local mode is needed for:
      // - Multi-thread mode: where graph capture and instantiation can happen concurrently across threads
      // - P2P pre-connect: when there is no warm-up, P2P pre-connect is done during graph capture.
      //   Since pre-connect calls hggcMalloc, we cannot use global capture mode
      HGGCCHECK(hggcStreamBeginCapture(args->streams[i], hggcStreamCaptureModeThreadLocal));
    }
  }

  // kernel launch
  int _iters = iters;
  if (rep_type > REP_DEFAULT) {
    _iters = 1;
  }
  for (int iter = 0; iter < _iters; iter++) {
    if (acu_profile) {
      if (target_proc == -1 || args->proc == target_proc) {
        hggcProfilerStart();
      }
      // for icn statistics sync.
      Barrier(args);
    }
    TESTCHECK(waitReady(args));
    if (agg_iters>1) PCCLCHECK(pcclGroupStart());
    for (int aiter = 0; aiter < agg_iters; aiter++) {
      TESTCHECK(startColl(args, type, op, root, in_place, iter*agg_iters+aiter));
    }
    if (agg_iters>1) PCCLCHECK(pcclGroupEnd());
    if (acu_profile && (target_proc == -1 || args->proc == target_proc))
      hggcProfilerStop();
  }

  if (hggcGraphLaunches >= 1) {
    // End hggc graph capture
    for (int i = 0; i < args->nGpus; i++) {
      HGGCCHECK(hggcStreamEndCapture(args->streams[i], graphs+i));
    }
    // Instantiate hggc graph
    for (int i = 0; i < args->nGpus; i++) {
      HGGCCHECK(hggcGraphInstantiate(graphExec+i, graphs[i], NULL, NULL, 0));
    }
    // Resync CPU, restart timing, launch hggc graph
    Barrier(args);

    if (elapsed_type == 0) {
      start = std::chrono::high_resolution_clock::now();
    } else if (elapsed_type == 1) {
      for (int i = 0; i < nGpus; i++) {
        HGGCCHECK(hggcEventRecord(args->startEvents[i], args->streams[i]));
      }
    }
    for (int l = 0; l < hggcGraphLaunches; l++) {
      for (int i = 0; i < args->nGpus; i++) {
        HGGCCHECK(hggcGraphLaunch(graphExec[i], args->streams[i]));
      }
    }
  }

  // record stop point
  if (elapsed_type == 1) {
    for (int i = 0; i < args->nGpus; i++) {
      HGGCCHECK(hggcEventRecord(args->stopEvents[i], args->streams[i]));
    }
  }

  int nIters = _iters * agg_iters;
  if (apiTimeUsec != NULL) {
    *apiTimeUsec = elapsedTime(start, nIters)*1.0E6;
  }
  // sync
  TESTCHECK(completeColl(args));

  double deltaSec = 0.0;
  if (elapsed_type == 0) {
    deltaSec = elapsedTime(start, nIters);
    Allreduce(args, &deltaSec, &deltaSec, 1, average, args->nThreads);
  } else if (elapsed_type == 1) {
    float deltaMsPerGpusF[args->nGpus];
    double deltaSecPerGpus[args->nGpus];
    for (int i = 0; i < args->nGpus; i++) {
      HGGCCHECK(hggcEventElapsedTime(&deltaMsPerGpusF[i], args->startEvents[i], args->stopEvents[i]));
      deltaSecPerGpus[i] = (double)(deltaMsPerGpusF[i] / nIters / 1e3);
    }
    Allreduce(args, &deltaSec, deltaSecPerGpus, args->nGpus, average, args->nThreads);
  }

  if (hggcGraphLaunches >= 1) {
    deltaSec = deltaSec / hggcGraphLaunches;
    //destroy hggc graph
    for (int i=0; i<args->nGpus; i++) {
      HGGCCHECK(hggcGraphExecDestroy(graphExec[i]));
      HGGCCHECK(hggcGraphDestroy(graphs[i]));
    }
  }
  *timeUsec = deltaSec*1.0E6;

  Barrier(args);
  return testSuccess;
}

testResult_t CheckCorrectness(struct threadArgs* args, pcclDataType_t type, pcclRedOp_t op, int root, int in_place, double* nWrongElt) {
  if (!datacheck) {
    return testSuccess;
  }

  Barrier(args);

  hggcGraph_t graphs[args->nGpus];
  hggcGraphExec_t graphExec[args->nGpus];

  static __thread int rep = 0;
  rep++;
  // Initialize sendbuffs, recvbuffs and expected
  TESTCHECK(args->collTest->initData(args, type, op, root, rep, in_place));

  if (hggcGraphLaunches >= 1) {
    // Begin hggc graph capture for data check
    for (int i=0; i<args->nGpus; i++) {
      HGGCCHECK(hggcStreamBeginCapture(args->streams[i], hggcStreamCaptureModeThreadLocal));
    }
  }

  //test validation in single itertion, should ideally be included into the multi-iteration run
  TESTCHECK(startColl(args, type, op, root, in_place, 0));

  if (hggcGraphLaunches >= 1) {
    // End hggc graph capture
    for (int i=0; i<args->nGpus; i++) {
      HGGCCHECK(hggcStreamEndCapture(args->streams[i], graphs+i));
    }
    // Instantiate hggc graph
    for (int i=0; i<args->nGpus; i++) {
      HGGCCHECK(hggcGraphInstantiate(graphExec+i, graphs[i], NULL, NULL, 0));
    }
    // Launch hggc graph
    for (int i=0; i<args->nGpus; i++) {
      HGGCCHECK(hggcGraphLaunch(graphExec[i], args->streams[i]));
    }
  }

  TESTCHECK(completeColl(args));

  if (hggcGraphLaunches >= 1) {
    //destroy hggc graph
    for (int i=0; i<args->nGpus; i++) {
      HGGCCHECK(hggcGraphExecDestroy(graphExec[i]));
      HGGCCHECK(hggcGraphDestroy(graphs[i]));
    }
  }

  *nWrongElt = 0;
  double nWrongEltPerThread = 0.0;
  TESTCHECK(CheckData(args, type, op, root, in_place, &nWrongEltPerThread, rep));

  //aggregate delta from all threads and procs
  Barrier(args);
  Allreduce(args, nWrongElt, &nWrongEltPerThread, 1, /*sum*/4, args->nThreads);

  return testSuccess;
}

void setupArgs(size_t size, pcclDataType_t type, struct threadArgs* args) {
  int nranks = args->nProcs*args->nGpus*args->nThreads;
  size_t count, sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset;

  count = size / wordSize(type);
  size_t eltSize = wordSize(type);
  args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset, &recvInplaceOffset, (size_t)count, eltSize, (size_t)nranks);

  args->nbytes = paramCount * wordSize(type);
  args->sendBytes = sendCount * wordSize(type);
  args->expectedBytes = recvCount * wordSize(type);
  args->sendInplaceOffset = sendInplaceOffset * wordSize(type);
  args->recvInplaceOffset = recvInplaceOffset * wordSize(type);
}

void getFloatStr(double value, size_t strSize, char* str) {
  if (!str || strSize == 0) return;
  // Need at least 4 bytes to store the overflow tag "OVF" plus the trailing '\0' (i.e., "OVF\0").
  // When the buffer is smaller, we can't report overflow safely, so return an empty string.
  if (strSize < 4) { str[0] = '\0'; return; }

  int ret, width = (int)strSize - 1;
  // Handle special cases: infinity and NaN
  if (isinf(value)) {
    const char* infStr = (value > 0) ? "INF" : "-INF";
    snprintf(str, strSize, "%*s", width, infStr);
    return;
  }
  if (isnan(value)) {
    snprintf(str, strSize, "%*s", width, "NAN");
    return;
  }

  int power = 0;
  for (uint64_t val = 1; value >= val; val *= 10) power++;

  if (power < width - 2)      ret = snprintf(str, strSize, "%*.2f", width, value);
  else if (power < width - 1) ret = snprintf(str, strSize, "%*.1f", width, value);
  else if (power < width + 1) ret = snprintf(str, strSize, "%*.0f", width, value);
  else if (width >= 8)        ret = snprintf(str, strSize, "%*.2e", width, value);
  else if (width >= 7)        ret = snprintf(str, strSize, "%*.1e", width, value);
  else                        ret = snprintf(str, strSize, "%*.0e", width, value);

  if (ret < 0 || (size_t)ret >= strSize) {
    snprintf(str, strSize, "%*s", (int)strSize - 1, "OVF");
  }
}

void printTimeTestResult(struct threadArgs* args, double *timeUsec, double *apiTimeUsec, double *algBw, double *busBw, double nWrongElt, double *avgAlgBw, double *avgBusBw, int n) {
    double sumTimeUsec = 0.0;
    double sumApiTimeUsec = 0.0;
    double sumAlgBw = 0.0;
    double sumBusBw = 0.0;
    for(int i = 0; i < n; i++) {
      sumTimeUsec += timeUsec[i];
      sumApiTimeUsec += apiTimeUsec[i];
      sumAlgBw += algBw[i];
      sumBusBw += busBw[i];
    }

    double avgTimeUsec = sumTimeUsec / n;
    *avgAlgBw = sumAlgBw / n;
    *avgBusBw = sumBusBw / n;
    char timeStr[8];
    getFloatStr(avgTimeUsec, sizeof(timeStr), timeStr);
    if (api_time) {
      double avgApiTimeUsec = sumApiTimeUsec / n;
      char apiTimeStr[8] = "";
      getFloatStr(avgApiTimeUsec, sizeof(apiTimeStr), apiTimeStr);
      PRINT("  %7s %8s %7.2f  %6.2f", timeStr, apiTimeStr, (*avgAlgBw), (*avgBusBw));
    } else {
      PRINT("  %7s %7.2f  %6.2f", timeStr, (*avgAlgBw), (*avgBusBw));
    }
    if (rep_type > REP_DEFAULT) {
      std::sort(busBw, busBw + n);
      double minBusBw = busBw[0];
      double maxBusBw = busBw[n - 1];
      double p90BusBw = busBw[int(0.9 * n) - 1];
      double p99BusBw = busBw[int(0.99 * n) -1];
      double varBusBw = 0.0;
      for(int i = 0; i < n; i++) {
        varBusBw += pow(busBw[i] - (*avgBusBw), 2);
      }
      varBusBw /= n;
      PRINT("  %6.2f  %6.2f  %6.2f  %6.2f  %6.2f", minBusBw, p90BusBw, p99BusBw, maxBusBw, varBusBw);
    }
    if (args->reportErrors) {
      PRINT("  %7g", nWrongElt);
    } else {
      PRINT("  %7s", "N/A");
    }
}

#ifndef DISABLE_HGPTI
testResult_t hgptiBegin(struct threadArgs* args) {
  if (args->thread == 0) HGPTICHECK(hgptiClean());
  Barrier(args);

  return testSuccess;
}

testResult_t hgptiEnd(struct threadArgs* args, double *timeUsec) {
  double *deltaSecPerGpusPerIter = (double*)calloc(args->nGpus*args->nThreads*iters, sizeof(double));
  double *deltaSecPerGpus = (double*)calloc(args->nGpus*args->nThreads, sizeof(double));
  if (args->thread == 0) {
    HGPTICHECK(hgptiFlush(deltaSecPerGpusPerIter));

    if (rep_type == REP_DEFAULT) {
      for (int devId = 0; devId < args->nGpus*args->nThreads; devId++) {
        for (int iterId = 0; iterId < iters; iterId++) {
          deltaSecPerGpus[devId] += deltaSecPerGpusPerIter[iters * devId + iterId];
        }
        deltaSecPerGpus[devId] /= iters;
      }
      Allreduce(args, timeUsec, deltaSecPerGpus, args->nGpus*args->nThreads, average, 1);
    } else {
      for (int iterId = 0; iterId < iters; iterId++) {
        for (int devId = 0; devId < args->nGpus*args->nThreads; devId++) {
          deltaSecPerGpus[devId] = deltaSecPerGpusPerIter[iters * devId + iterId];
        }
        Allreduce(args, &timeUsec[iterId], deltaSecPerGpus, args->nGpus*args->nThreads, average, 1);
      }
    }
  }

  Barrier(args);
  free(deltaSecPerGpusPerIter);
  free(deltaSecPerGpus);
  return testSuccess;
}
#endif

testResult_t TimeTestPerSize(struct threadArgs* args, pcclDataType_t type, size_t size, const char* typeName, pcclRedOp_t op, const char* opName, int root, struct _IO_FILE *busbw_file) {
  if (size <= 0) return testSuccess;

  setupArgs(size, type, args);
  size_t output_buff_size = max(args->sendBytes, args->expectedBytes);
  char rootName[100];
  sprintf(rootName, "%5i", root);
  PRINT("%12li  %12li  %8s  %6s  %5s", max(args->sendBytes, args->expectedBytes), args->nbytes / wordSize(type), typeName, opName, rootName);
  int _iters = 1;
  if (rep_type > REP_DEFAULT){
    _iters = iters;
  }
  double *timeUsec = (double*)malloc(_iters * sizeof(double));
  double *apiTimeUsec = (double*)malloc(_iters * sizeof(double));
  double *algBw = (double*)malloc(_iters * sizeof(double));
  double *busBw = (double*)malloc(_iters * sizeof(double));
  double nWrongElt = 0.0;

  for (int _bk = 1; _bk >= 0; _bk--) {
    if (test_buffer_kind == 2 || _bk == test_buffer_kind) {
      int in_place = (_bk == 0) ? 1 : 0;
      if (rep_type == REP_DETAIL) {
        fprintf(busbw_file, "\n%li %s busBw: ", output_buff_size, in_place ? "inplace" : "out-of-place");
      }
      // Sync
      if (!single_test) {
        TESTCHECK(startColl(args, type, op, root, in_place, 0));
        TESTCHECK(completeColl(args));
      }

#ifndef DISABLE_HGPTI
      if (elapsed_type == 2) HGPTICHECK(hgptiBegin(args));
#endif
      for(int i = 0; i < _iters; i++) {
        timeUsec[i] = 0.0;
        apiTimeUsec[i] = 0.0;
        TESTCHECK(BenchTime(args, type, op, root, in_place, timeUsec+i, apiTimeUsec+i));
      }
#ifndef DISABLE_HGPTI
      if (elapsed_type == 2) HGPTICHECK(hgptiEnd(args, timeUsec));
#endif

      for (int i = 0; i < _iters; i++) {
        algBw[i] = 0.0; busBw[i] = 0.0;
        if (timeUsec[i] == 0.0) continue;
        args->collTest->getBw(args->nbytes/wordSize(type), wordSize(type), timeUsec[i]/1.0E6,
                              &algBw[i], &busBw[i], args->nProcs*args->nThreads*args->nGpus);
        if (rep_type == REP_DETAIL) {
          if (i % 10 == 0) {
            fprintf(busbw_file, "\n  ");
          }
          fprintf(busbw_file, "  %6.2f", busBw[i]);
          fflush(busbw_file);
        }
      }

      if (datacheck) {
        TESTCHECK(CheckCorrectness(args, type, op, root, in_place, &nWrongElt));
      }
      double avgAlgBw = 0.0, avgBusBw = 0.0;
      if (is_main_thread) {
        printTimeTestResult(args, timeUsec, apiTimeUsec, algBw, busBw, nWrongElt, &avgAlgBw, &avgBusBw, _iters);
        args->bw[0] += avgBusBw;
        args->bw_count[0]++;
      }
    }
  }
  PRINT("\n");
  free(timeUsec);
  free(apiTimeUsec);
  free(algBw);
  free(busBw);
  return testSuccess;
}

testResult_t TimeTest(struct threadArgs* args, pcclDataType_t type, const char* typeName, pcclRedOp_t op, const char* opName, int root) {
  // Warm-up for large size
  setupArgs(args->maxbytes, type, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    TESTCHECK(startColl(args, type, op, root, 0, iter));
  }
  TESTCHECK(completeColl(args));

  // Warm-up for small size
  setupArgs(args->minbytes, type, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    TESTCHECK(startColl(args, type, op, root, 0, iter));
  }
  TESTCHECK(completeColl(args));

  // create hggc events and warm up
  if (elapsed_type == 1) {
    args->startEvents = (hggcEvent_t*)malloc(nGpus * sizeof(hggcEvent_t));
    args->stopEvents = (hggcEvent_t*)malloc(nGpus * sizeof(hggcEvent_t));
    for (int i = 0; i < nGpus; i++) {
      int hggcDev;
      PCCLCHECK(pcclCommCuDevice(args->comms[i], &hggcDev));
      HGGCCHECK(hggcSetDevice(hggcDev));
      HGGCCHECK(hggcEventCreate(args->startEvents+i));
      HGGCCHECK(hggcEventCreate(args->stopEvents+i));
    }
    for (int i = 0; i < args->nGpus; i++) {
      HGGCCHECK(hggcEventRecord(args->startEvents[i], args->streams[i]));
      HGGCCHECK(hggcEventRecord(args->stopEvents[i], args->streams[i]));
    }
  }

  struct _IO_FILE *busbw_file = NULL;
  if (rep_type == REP_DETAIL) {
    busbw_file = fopen("./busbw_raw_data.log", "w");
  }

  // Benchmark
  if (size_list_num == 0) {
    for (size_t size = args->minbytes; size<=args->maxbytes; size = ((args->stepfactor > 1) ? size*args->stepfactor : size+args->stepbytes)) {
      TimeTestPerSize(args, type, size, typeName, op, opName, root, busbw_file);
    }
  } else {
    for (int i = 0; i < size_list_num; i++) {
      ssize_t size = listBytes[i];
      TimeTestPerSize(args, type, size, typeName, op, opName, root, busbw_file);
    }
  }

  if (elapsed_type == 1) {
    if (args->startEvents) free(args->startEvents);
    if (args->stopEvents) free(args->stopEvents);
  }

  return testSuccess;
}

testResult_t threadRunTests(struct threadArgs* args) {
  // Set device to the first of our GPUs. If we don't do that, some operations
  // will be done on the current GPU (by default : 0) and if the GPUs are in
  // exclusive mode those operations will fail.
  HGGCCHECK(hggcSetDevice(args->gpus[0]));

  if (!init_only) {
    TESTCHECK(pcclTestEngine.runTest(args, pcclroot, (pcclDataType_t)pccltype, test_typenames[pccltype], (pcclRedOp_t)pcclop, test_opnames[pcclop]));
  }
  return testSuccess;
}

testResult_t threadInit(struct threadArgs* args) {
  char hostname[1024];
  getHostName(hostname, 1024);
  int nranks =  args->nProcs*args->nThreads*args->nGpus;

  //set main thread again
  is_main_thread = (args->proc == 0 && args->thread == 0) ? 1 : 0;

  PCCLCHECK(pcclGroupStart());
  for (int i=0; i<args->nGpus; i++) {
    int rank = args->proc*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    PCCLCHECK(pcclCommInitRank(args->comms+i, nranks, args->pcclIds[0], rank));
  }
  PCCLCHECK(pcclGroupEnd());

  void **sendRegHandles = NULL;
  void **recvRegHandles = NULL;
  if (local_register != 0) {
    PCCLCHECK(pcclGroupStart());
    sendRegHandles = (local_register) ? (void **)malloc(sizeof(*sendRegHandles)*args->nGpus) : NULL;
    recvRegHandles = (local_register) ? (void **)malloc(sizeof(*recvRegHandles)*args->nGpus) : NULL;
    args->sendRegHandles = sendRegHandles;
    args->recvRegHandles = recvRegHandles;
    for (int i=0; i<args->nGpus; i++) {
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,0,0)
      if (local_register == SYMMETRIC_REGISTER) {
        PCCLCHECK(pcclCommWindowRegister(args->comms[i], args->sendbuffs[i], args->maxbytes, (pcclWindow_t*)&sendRegHandles[i],PCCL_WIN_COLL_SYMMETRIC));
        PCCLCHECK(pcclCommWindowRegister(args->comms[i], args->recvbuffs[i], args->maxbytes, (pcclWindow_t*)&recvRegHandles[i],PCCL_WIN_COLL_SYMMETRIC));
      } else
#endif
      {
        if (local_register) PCCLCHECK(pcclCommRegister(args->comms[i], args->sendbuffs[i], args->maxbytes, &sendRegHandles[i]));
        if (local_register) PCCLCHECK(pcclCommRegister(args->comms[i], args->recvbuffs[i], args->maxbytes, &recvRegHandles[i]));
      }
    }
    PCCLCHECK(pcclGroupEnd());
  }

#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
  if (deviceImpl > 0) {
    pcclDevCommRequirements reqs = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    TESTCHECK(pcclTestEngine.getDevCommRequirements(deviceImpl, &reqs, args->comms[0]));
    PCCLCHECK(pcclGroupStart());
    for (int i = 0; i < args->nGpus; i++) {
      PCCLCHECK((pcclResult_t)pcclDevCommCreate(args->comms[i], &reqs, args->devComms+i));
    }
    PCCLCHECK(pcclGroupEnd());
  }
#endif

  TESTCHECK(threadRunTests(args));

#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
  if (deviceImpl > 0) {
    for (int i = 0; i < args->nGpus; i++) {
      PCCLCHECK((pcclResult_t)pcclDevCommDestroy(args->comms[i], args->devComms+i));
    }
  }
#endif

  for (int i=0; i<args->nGpus; i++) {
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,0,0)
    if (local_register == SYMMETRIC_REGISTER) {
      PCCLCHECK(pcclCommWindowDeregister(args->comms[i], (pcclWindow_t)sendRegHandles[i]));
      PCCLCHECK(pcclCommWindowDeregister(args->comms[i], (pcclWindow_t)recvRegHandles[i]));
    } else
#endif
    {
      if (local_register) PCCLCHECK(pcclCommDeregister(args->comms[i], sendRegHandles[i]));
      if (local_register) PCCLCHECK(pcclCommDeregister(args->comms[i], recvRegHandles[i]));
    }
    PCCLCHECK(pcclCommDestroy(args->comms[i]));
  }
  return testSuccess;
}

void* threadLauncher(void* thread_) {
  struct testThread* thread = (struct testThread*)thread_;
  thread->ret = thread->func(&thread->args);
  return NULL;
}
testResult_t threadLaunch(struct testThread* thread) {
  pthread_create(&thread->thread, NULL, threadLauncher, thread);
  return testSuccess;
}

testResult_t AllocateBuffs(int dev_id, void **sendbuffs_origin, void **sendbuff, size_t sendBytes,
                           void **recvbuffs_origin, void **recvbuff, size_t recvBytes,
                           void **expected, size_t nbytes, int nranks) {
    PCCLCHECK(pcclMemAlloc(sendbuffs_origin, nbytes + sendbuff_offset));
    PCCLCHECK(pcclMemAlloc(recvbuffs_origin, nbytes + recvbuff_offset));
    *sendbuff = (void*)((uintptr_t)(*sendbuffs_origin) + (uintptr_t)(dev_id % 2 == 0 ? sendbuff_offset : 0));
    *recvbuff = (void*)((uintptr_t)(*recvbuffs_origin) + (uintptr_t)(dev_id % 2 == 0 ? recvbuff_offset : 0));
    if (datacheck) PCCLCHECK(pcclMemAlloc(expected, recvBytes));
    return testSuccess;
}

testResult_t run(); // Main function

int main(int argc, char* argv[]) {
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  pcclGetVersion(&test_pcclVersion);

  test_typenum = PCCL_NUM_BASE_TYPES;

  // Parse args
  int longindex;
  static struct option longopts[] = {
    {"nthreads", required_argument, 0, 't'},
    {"ngpus", required_argument, 0, 'g'},
    {"minbytes", required_argument, 0, 'b'},
    {"maxbytes", required_argument, 0, 'e'},
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"iters", required_argument, 0, 'n'},
    {"agg_iters", required_argument, 0, 'm'},
    {"warmup_iters", required_argument, 0, 'w'},
    {"init_type", required_argument, 0, 'p'},
    {"check", required_argument, 0, 'c'},
    {"op", required_argument, 0, 'o'},
    {"datatype", required_argument, 0, 'd'},
    {"root", required_argument, 0, 'r'},
    {"blocking", required_argument, 0, 'z'},
    {"init_blocking_timeout", required_argument, 0, 'T'},
    {"hggcgraph", required_argument, 0, 'G'},
    {"average", required_argument, 0, 'a'},
    {"local_register", required_argument, 0, 'R'},
    {"cta_policy", required_argument, 0, 'x'},
    {"single_test", required_argument, 0, 's'},
    {"test_buffer_kind", required_argument, 0, 'k'},
    {"elapsed_type", required_argument, 0, OPT_ELAPSED_TYPE},
    {"api_time", optional_argument, 0, OPT_API_TIME},
    {"init_blocking", required_argument, 0, OPT_INIT_BLOCKING},
    {"random_size_num", required_argument, 0, OPT_RANDOM_SIZE_NUM},
    {"sizelist", required_argument, 0, OPT_SIZE_LIST},
    {"init_only", required_argument, 0, OPT_INIT_ONLY},
    {"core_dump", required_argument, 0, OPT_CORE_DUMP},
    {"acu_profile", required_argument, 0, OPT_ACU_PROFILE},
    {"target_process", required_argument, 0, OPT_TARGET_PROCESS},
    {"abnormal_mode", required_argument, 0, OPT_ABNORMAL_MODE},
    {"buffer_offset", required_argument, 0, OPT_BUFFER_OFFSET},
    {"report_type", required_argument, 0, OPT_REPORT_TYPE},
    {"report_check_filename", required_argument, 0, OPT_REPORT_CHECK_FILENAME},
    {"device_implementation", required_argument, 0, 'D'},
    {"device_cta_count", required_argument, 0, 'V'},
    {"help", no_argument, 0, 'h'},
    {}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "t:g:b:e:i:f:n:m:w:p:c:o:d:r:z:T:hG:a:R:x:s:k:D:V:", longopts, &longindex);

    if (c == -1)
      break;

    switch(c) {
      case 't':
        nThreads = strtol(optarg, NULL, 0);
        break;
      case 'g':
        nGpus = strtol(optarg, NULL, 0);
        break;
      case 'b':
        minBytes = (size_t)parsesize(optarg);
        break;
      case 'e':
        maxBytes = (size_t)parsesize(optarg);
        break;
      case 'i':
        stepBytes = strtol(optarg, NULL, 0);
        break;
      case 'f':
        stepFactor = strtol(optarg, NULL, 0);
        break;
      case 'n':
        iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'm':
        agg_iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'w':
        warmup_iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'p':
        init_type = (int)strtol(optarg, NULL, 0);
        break;
      case 'c':
        datacheck = (int)strtol(optarg, NULL, 0);
        break;
      case 'o':
        pcclop = pcclstringtoop(optarg);
        break;
      case 'd':
        pccltype = pcclstringtotype(optarg);
        break;
      case 'r':
        pcclroot = strtol(optarg, NULL, 0);
        break;
      case 'z':
        blocking_coll = strtol(optarg, NULL, 0);
        break;
      case 'T':
        init_blocking_timeout = (reportType_t)strtol(optarg, NULL, 0);
        break;
      case 'G':
        hggcGraphLaunches = strtol(optarg, NULL, 0);
        break;
      case 'a':
        average = (int)strtol(optarg, NULL, 0);
        if (average < 0 || average > 3) {
          printf("abnormal average: %d, use 2(MIN) instead\n", average);
          average = 2;
        }
        break;
      case 'R':
        local_register = (int)strtol(optarg, NULL, 0);
        break;
      case 'x':
        ctaPolicy = (int)strtol(optarg, NULL, 0);
        break;
      case 's':
        single_test = (int)strtol(optarg, NULL, 0);
        break;
      case 'k':
        test_buffer_kind = (int)strtol(optarg, NULL, 3);
        break;
      case OPT_CORE_DUMP:
        core_dump = (int)strtol(optarg, NULL, 0);
        break;
      case OPT_ELAPSED_TYPE:
        elapsed_type = (int)strtol(optarg, NULL, 0);
        break;
      case OPT_ACU_PROFILE:
        acu_profile = (int)strtol(optarg, NULL, 0);
        break;
      case OPT_REPORT_TYPE:
        rep_type = (reportType_t)strtol(optarg, NULL, 0);
        break;
      case OPT_INIT_ONLY:
        init_only = (reportType_t)strtol(optarg, NULL, 0);
        break;
      case OPT_INIT_BLOCKING:
        init_blocking = (reportType_t)strtol(optarg, NULL, 0);
        break;
      case OPT_TARGET_PROCESS:
        target_proc = (reportType_t)strtol(optarg, NULL, 0);
        break;
      case OPT_SIZE_LIST:
        parsesizelist(optarg);
        break;
      case OPT_REPORT_CHECK_FILENAME:
        report_check_filename = optarg;
        break;
      case 'D':
        deviceImpl = (int)strtol(optarg, NULL, 0);
        break;
      case 'V':
        deviceCtaCount = (int)strtol(optarg, NULL, 0);
        deviceCtaCountSet = true;
        if (deviceCtaCount <= 0 || deviceCtaCount > 128) {
          fprintf(stderr, "device_cta_count must be positive and less than 128, got %d. "
                  "Using default value 16.\n", deviceCtaCount);
          deviceCtaCount = 16;
        }
        break;
      case OPT_API_TIME:
        api_time = true;
        break;
      case OPT_BUFFER_OFFSET:
        sendbuff_offset = strtol(optarg, NULL, 0);
        recvbuff_offset = strtol(optarg, NULL, 0);
        break;
      case OPT_RANDOM_SIZE_NUM:
        randomSizeNum = strtol(optarg, NULL, 0);
        break;
      case OPT_ABNORMAL_MODE:
        abnormal_mode = (int)strtol(optarg, NULL, 0);
        break;
      case 'h':
      default:
        if (c != 'h') printf("invalid option '%c'\n", c);
        printf("USAGE: %s \n\t"
            "[-t,--nthreads <num threads>] \n\t"
            "[-g,--ngpus <gpus per thread>] \n\t"
            "[-b,--minbytes <min size in bytes>] \n\t"
            "[-e,--maxbytes <max size in bytes>] \n\t"
            "[-i,--stepbytes <increment size>] \n\t"
            "[-f,--stepfactor <increment factor>] \n\t"
            "[-n,--iters <iteration count>] \n\t"
            "[-m,--agg_iters <aggregated iteration count>] \n\t"
            "[-w,--warmup_iters <warmup iteration count>] \n\t"
            "[-p,--init_type <0/1/2>] \n\t"
            "[-c,--check <0/1>] \n\t"
            "[-o,--op <sum/prod/min/max/avg/mulsum/all>] \n\t"
            "[-d,--datatype <pccltype/all>] \n\t"
            "[-r,--root <root>] \n\t"
            "[-z,--blocking <0/1>] \n\t"
            "[-T,--init_blocking_timeout <seconds>] \n\t"
            "[-G,--hggcgraph <num graph launches>] \n\t"
            "[-a,--average <0/1/2/3> Report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] \n\t"
            "[-R,--local_register <0/1/2> Enable local (1) or symmetric (2) buffer registration on send/recv buffers (default: disable (0))] \n\t"
            "[-x,--cta_policy <0/1/2> Set CTA policy (PCCL_CTA_POLICY_DEFAULT (0), PCCL_CTA_POLICY_EFFICIENCY (1), PCCL_CTA_POLICY_ZERO (2)) (default: do not set)] \n\t"
            "[-s,--single_test <0/1>] \n\t"
            "[-k,--test_buffer_kind <0/1/2>] \n\t"
            "[--elapsed_type <0/1/2>] Timing source: 0=CPU,1=event,2=hgpti \n\t"
            "[--api_time] Report API overhead time \n\t"
            "[--init_only <0/1>] Initialization only, skip tests \n\t"
            "[--init_blocking <0/1>] Enable blocking initialization \n\t"
            "[--abnormal_mode <0/1/2/3>] Abnormal test mode \n\t"
            "[--target_process <0/1/...>] Target process for ACU profiling \n\t"
            "[--acu_profile <0/1>] Enable Asight Compute profiling \n\t"
            "[--buffer_offset <offset bytes>] Send/recv buffer offset \n\t"
            "[--random_size_num <N>] Generate N random sizes \n\t"
            "[--sizelist <size>[,size]] Comma-separated size list \n\t"
            "[--core_dump <0/1>] Enable core dump generation \n\t"
            "[--report_type <0/1/2>] Report detail level \n\t"
            "[--report_check_filename <filename string>] Output file for error reports \n\t"
            "[-D,--device_implementation <impl number>] Enable device implementation (default: 0; requires -R 2 if > 0) \n\t"
            "[-V,--device_cta_count <number>] Set number of CTAs for device implementation (default: 16) \n\t"
            "[-h,--help]\n",
            basename(argv[0]));
        return 0;
    }
  }
  if(core_dump) {
      // enable core dump
      struct rlimit core_limits;
      core_limits.rlim_cur = core_limits.rlim_max = RLIM_INFINITY;
      setrlimit(RLIMIT_CORE, &core_limits);
  }
  if (rep_type != REP_DEFAULT) api_time = false;    // api_time only allowed for REP_DEFAULT

  size_t elemSize = wordSize((pcclDataType_t)pccltype);
  if (sendbuff_offset != 0 && sendbuff_offset % elemSize != 0) {
    sendbuff_offset = (sendbuff_offset / elemSize) * elemSize;
    printf("sendbuffer offset must be divisible by the size of type: %zu, rounded down\n", elemSize);
  }
  if (recvbuff_offset != 0 && recvbuff_offset % elemSize != 0) {
    recvbuff_offset = (recvbuff_offset / elemSize) * elemSize;
    printf("recvbuffer offset must be divisible by the size of type: %zu, rounded down\n", elemSize);
  }
  if (maxBytes < minBytes) {
    printf("maxbytes: %lu can't be less than minbytes: %lu\n", maxBytes, minBytes);
    exit(EXIT_FAILURE);
  }
  if (deviceImpl > 0) {
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
    if (test_pcclVersion < PCCL_VERSION(2,2,0)) {
      fprintf(stderr, "device implementation (--device_implementation > 0) requires PCCL >= 2.2.0, "
              "but is running with %d.\n", test_pcclVersion);
      exit(EXIT_FAILURE);
    }
    if (!pcclTestEngine.getDevCommRequirements || !pcclTestEngine.runCollDevice) {
      fprintf(stderr, "device implementation %d is not supported by this test\n", deviceImpl);
      exit(EXIT_FAILURE);
    }
#else
    fprintf(stderr, "device implementation (--device_implementation > 0) requires pccl-tests to be "
            "compiled against PCCL >= 2.2.0\n");
    exit(EXIT_FAILURE);
#endif
  }
  if (deviceImpl > 0 && (local_register != SYMMETRIC_REGISTER)) {
    fprintf(stderr, "device implementation (--device_implementation > 0) requires enabling symmetric memory registration (-R 2)\n");
    exit(EXIT_FAILURE);
  }
  if (acu_profile) blocking_coll = 1;

#ifdef MPI_SUPPORT
  MPI_Init(&argc, &argv);
#endif
  return run();
}

int getIntFromEnv(const char *prop, int defaultVal) {
  char *env = getenv(prop);
  return env ? atoi(env) : defaultVal;
}

testResult_t run() {
  int totalProcs = 1, proc = 0, pcclProcs = 1, pcclProc = 0, color = 0;
  int localRank = 0;
  char hostname[1024];
  getHostName(hostname, 1024);

#ifdef MPI_SUPPORT
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  char* needSleepStr = getenv("NCCL_TEST_SLEEP");
  int needSleep = needSleepStr ? atoi(needSleepStr) : 0;
  if (proc == 0 && needSleep) {
    printf("[DEBUG] Proc 0 (PID: %d) is paused because NCCL_TEST_SLEEP is set.\n To continue in GDB:\n 1. Switch to the 'run()' frame (e.g., 'frame <N>').\n 2. Execute 'set var flag = 1'.\n", getpid());
  }
  volatile int flag = 0;
  // Proc 0 blocks on the flag, which must be manually changed in a debugger (e.g., GDB). All other procs will block in MPI_Allgather waiting for proc 0.
  while (proc==0 && needSleep && flag==0) {
    sleep(5);
  }
  uint64_t hostHashs[totalProcs];
  hostHashs[proc] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int p=0; p<totalProcs; p++) {
    if (p == proc) break;
    if (hostHashs[p] == hostHashs[proc]) localRank++;
  }

  char *maskstr = getenv("NCCL_TESTS_SPLIT_MASK");
  uint64_t mask = maskstr ? strtoul(maskstr, NULL, 16) : 0;
  MPI_Comm mpi_comm;
  color = proc & mask;
  MPI_Comm_split(MPI_COMM_WORLD, color, proc, &mpi_comm);
  MPI_Comm_size(mpi_comm, &pcclProcs);
  MPI_Comm_rank(mpi_comm, &pcclProc);
#else
  totalProcs = pcclProcs = getIntFromEnv("WORLD_SIZE", 1);
  proc = pcclProc = getIntFromEnv("RANK", 0);
  barrierInit(proc, totalProcs);
#endif
  is_main_thread = (proc == 0) ? 1 : 0;

  PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld offset <%ld>/<%ld> step: %ld(%s) warmup iters: %d "
      "iters: %d redop: %s validation: %d single_test: %d test_buffer_kind: %s elapsed_type: %s average: %s register: %d\n",
      nThreads, nGpus, minBytes, maxBytes, sendbuff_offset, recvbuff_offset, (randomSizeNum>0)?randomSizeNum:((stepFactor > 1)?stepFactor:stepBytes),
      (randomSizeNum>0)?"random":((stepFactor > 1)?"factor":"bytes"), warmup_iters, iters, pcclop == -1 ? "all" : test_opnames[pcclop], datacheck, single_test,
      (test_buffer_kind == 0) ? "inplace" : (test_buffer_kind == 1) ? "outplace" : "both",
      (elapsed_type == 0) ? "cpu_time" : (elapsed_type == 1) ? "event_time" : (elapsed_type == 2) ? "kernel_time" : "None", averageStr[average], local_register);
  if (blocking_coll) PRINT("# Blocking Enabled: wait for completion and barrier after each collective \n");
  if (init_type == INIT_PARALLEL) PRINT("# Parallel Init Enabled: threads call into PcclInitRank concurrently \n");
  PRINT("#\n");

  PRINT("# Using devices\n");
#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  int gpu0 = getIntFromEnv("NCCL_TESTS_DEVICE", -1);
  for (int i=0; i<nThreads*nGpus; i++) {
    int hggcDev = (gpu0 != -1 ? gpu0 : localRank*nThreads*nGpus) + i;
    int rank = proc*nThreads*nGpus+i;
    hggcDeviceProp prop;
    HGGCCHECK(hggcSetDevice(hggcDev));
    HGGCCHECK(hggcGetDeviceProperties(&prop, hggcDev));
    len += snprintf(line+len, MAX_LINE-len, "#   Rank %2d Group %2d Pid %7d on %10s device %2d [0x%02x] %s\n",
                    rank, color, getpid(), hostname, hggcDev, prop.pciBusID, prop.name);
  }

#ifdef MPI_SUPPORT
  char *lines = (proc == 0) ? (char *)malloc(totalProcs*MAX_LINE) : NULL;
  // Gather all output in rank order to root (0)
  MPI_Gather(line, MAX_LINE, MPI_BYTE, lines, MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (proc == 0) {
    for (int p = 0; p < totalProcs; p++)
      PRINT("%s", lines+MAX_LINE*p);
    free(lines);
  }
#else
  PRINT("%s", line);
#endif

  if (randomSizeNum > 0) {
    size_list_num = randomSizeNum;
    if (proc == 0) {
      std::random_device rd;
      long random_seed = rd();
      std::mt19937 gen(random_seed);
      std::uniform_int_distribution<size_t> distr(minBytes, maxBytes);
      for (int i = 0; i < randomSizeNum; i++) {
        ssize_t size = distr(gen);
        listBytes[i] = size;
      }
    }
#if MPI_SUPPORT
    MPI_Bcast(listBytes, randomSizeNum*sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  }

  int nRanks = pcclProcs*nThreads*nGpus;
  int nIds = init_type == INIT_SCALABLE ? std::max(2, nRanks/8) : 1;
  pcclUniqueId pcclIds[nIds];
#ifdef MPI_SUPPORT
  pcclUniqueId pcclIdsExchange[nRanks];
  if (init_type == INIT_SCALABLE) {
    for (int i=0; i<nGpus*nThreads; i++) {
      int rank = pcclProc*nThreads*nGpus+i;
      if (rankHasRoot(rank, nRanks, nIds)) {
        pcclGetUniqueId(pcclIdsExchange+rank);
      }
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pcclIdsExchange, nGpus*nThreads*sizeof(pcclUniqueId), MPI_BYTE, MPI_COMM_WORLD);
  } else {
    if (pcclProc == 0) PCCLCHECK(pcclGetUniqueId(pcclIds));
    MPI_Bcast(pcclIds, sizeof(pcclUniqueId), MPI_BYTE, 0, mpi_comm);
  }
  MPI_Barrier(MPI_COMM_WORLD); // Ensure Bcast is complete for HCOLL
  if (init_type == INIT_SCALABLE) {
    for (int id=0, rank=0; id<nIds && rank<nRanks; rank++) {
      if (rankHasRoot(rank, nRanks, nIds)) {
        memcpy(pcclIds+id++, pcclIdsExchange+rank, sizeof(pcclUniqueId));
      }
    }
  }
#else
  PCCLCHECK(pcclGetUniqueId(pcclIds));
#endif
  int gpus[nGpus*nThreads];
  hggcStream_t streams[nGpus*nThreads];
  void* sendbuffs_origin[nGpus*nThreads];
  void* recvbuffs_origin[nGpus*nThreads];
  void* sendbuffs[nGpus*nThreads];
  void* recvbuffs[nGpus*nThreads];
  void* expected[nGpus*nThreads];
  size_t sendBytes, recvBytes;
  size_t kerNum = 1;

  pcclTestEngine.getBuffSize(&sendBytes, &recvBytes, &kerNum, (size_t)maxBytes, (size_t)pcclProcs*nGpus*nThreads);

  if (elapsed_type == 2) {
#ifdef DISABLE_HGPTI
    printf("hgpti not supported!");
    exit(EXIT_FAILURE);
#else
    // set max device number per node for every proc now (16*810E)
    HGPTICHECK(hgptiInit(iters, kerNum, 16));
    blocking_coll = 1;
#endif
  }

  gpu0 = getIntFromEnv("NCCL_TESTS_DEVICE", -1);
  for (int i=0; i<nGpus*nThreads; i++) {
    gpus[i] = (gpu0 != -1 ? gpu0 : localRank*nThreads*nGpus) + i;
    HGGCCHECK(hggcSetDevice(gpus[i]));
    TESTCHECK(AllocateBuffs(gpus[i], sendbuffs_origin+i, sendbuffs+i, sendBytes, recvbuffs_origin+i, recvbuffs+i, recvBytes, expected+i, (size_t)maxBytes, pcclProcs*nThreads*nGpus));
    HGGCCHECK(hggcStreamCreateWithFlags(streams+i, hggcStreamNonBlocking));

    int archMajor, archMinor;
    HGGCCHECK(hggcDeviceGetAttribute(&archMajor, hggcDevAttrComputeCapabilityMajor, gpus[i]));
    HGGCCHECK(hggcDeviceGetAttribute(&archMinor, hggcDevAttrComputeCapabilityMinor, gpus[i]));
    minHggcArch = std::min(minHggcArch, 10 * archMajor + archMinor);
    HGGCCHECK(hggcDeviceGetAttribute(&deviceMpCount, hggcDevAttrMultiProcessorCount, gpus[i]));
  }

  pcclConfig_t config = PCCL_CONFIG_INITIALIZER;
  if (init_blocking) config.blocking = 1;
  else config.blocking = 0;

  if (ctaPolicy >=0) config.CTAPolicy = ctaPolicy;

#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &minHggcArch, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
  // Default: one CTA per CU on PPU 1.5, 16 otherwise.
  if (deviceImpl == 2 && !deviceCtaCountSet && minHggcArch == 89 && deviceMpCount > 0)
    deviceCtaCount = std::min(deviceMpCount, 128);
#if HAVE_BF16
  test_typenum++; // bfloat16
#endif
#if HAVE_FP8
  if (PCCL_VERSION_CODE >= PCCL_VERSION(1,7,0) && test_pcclVersion >= PCCL_VERSION(1,7,0)) {
    test_typenum += 2; // fp8 e4m3,e5m2
    if (minHggcArch < 89) { // Filter out fp8 on pre-PPU 1.5 hardware
      int n = 0;
      for (int i = 0; i < test_typenum; i++) {
        if (!(test_types[i] == pcclFloat8e4m3 || test_types[i] == pcclFloat8e5m2)) {
          test_types[n] = test_types[i];
          test_typenames[n] = test_typenames[i];
          n += 1;
        }
      }
      test_typenum = n;
    }
  }
#endif

  //if parallel init is not selected, use main thread to initialize PCCL
  pcclComm_t* comms = (pcclComm_t*)malloc(sizeof(pcclComm_t)*nThreads*nGpus);
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
  pcclDevComm* devComms = (pcclDevComm*)calloc(nThreads*nGpus, sizeof(pcclDevComm));
#endif
  void **sendRegHandles = NULL;
  void **recvRegHandles = NULL;
  if (init_type != INIT_PARALLEL) {
    if (pcclProcs == 1) {
      PCCLCHECK(pcclCommInitAll(comms, nGpus*nThreads, gpus));
    } else {
      PCCLCHECK(pcclGroupStart());
      for (int i=0; i<nGpus*nThreads; i++) {
        HGGCCHECK(hggcSetDevice(gpus[i]));
        if (init_type == INIT_CONFIG) {
          PCCLCHECK(pcclCommInitRankConfig(comms+i, pcclProcs*nThreads*nGpus, pcclIds[0], pcclProc*nThreads*nGpus+i, &config));
        } else if (init_type == INIT_SCALABLE) {
          PCCLCHECK(pcclCommInitRankScalable(comms+i, pcclProcs*nThreads*nGpus, pcclProc*nThreads*nGpus+i, nIds, pcclIds, &config));
        }
      }
      PCCLCHECK(pcclGroupEnd());
    }

    if (local_register != 0) {
      for (int i=0; i<nGpus*nThreads; i++) {
        pcclResult_t initRes;
        if(init_blocking == 0) while (pcclCommGetAsyncError(comms[i], &initRes), initRes == pcclInProgress) {}
      }
      PCCLCHECK(pcclGroupStart());
      sendRegHandles = (local_register) ? (void **)malloc(sizeof(*sendRegHandles)*nThreads*nGpus) : NULL;
      recvRegHandles = (local_register) ? (void **)malloc(sizeof(*recvRegHandles)*nThreads*nGpus) : NULL;
      for (int i=0; i<nGpus*nThreads; i++) {
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,0,0)
        if (local_register == SYMMETRIC_REGISTER) {
          PCCLCHECK(pcclCommWindowRegister(comms[i], sendbuffs[i], maxBytes, (pcclWindow_t*)&sendRegHandles[i], PCCL_WIN_COLL_SYMMETRIC));
          PCCLCHECK(pcclCommWindowRegister(comms[i], recvbuffs[i], maxBytes, (pcclWindow_t*)&recvRegHandles[i], PCCL_WIN_COLL_SYMMETRIC));
        } else
#endif
        {
          if (local_register) PCCLCHECK(pcclCommRegister(comms[i], sendbuffs[i], maxBytes, &sendRegHandles[i]));
          if (local_register) PCCLCHECK(pcclCommRegister(comms[i], recvbuffs[i], maxBytes, &recvRegHandles[i]));
        }
      }
      PCCLCHECK(pcclGroupEnd());
    }

#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
    if (deviceImpl > 0) {
      pcclDevCommRequirements reqs = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      TESTCHECK(pcclTestEngine.getDevCommRequirements(deviceImpl, &reqs, comms[0]));
      PCCLCHECK(pcclGroupStart());
      for (int i = 0; i < nGpus * nThreads; i++) {
        PCCLCHECK((pcclResult_t)pcclDevCommCreate(comms[i], &reqs, devComms+i));
      }
      PCCLCHECK(pcclGroupEnd());
    }
#endif
  }

  int errors[nThreads];
  double bw[nThreads];
  int bw_count[nThreads];
  for (int t=0; t<nThreads; t++) {
    bw[t] = 0.0;
    errors[t] = bw_count[t] = 0;
  }

  fflush(stdout);
  print_header(rep_type);

  int* barrier = (int*)calloc(2, sizeof(int));

  struct testThread threads[nThreads];
  memset(threads, 0, sizeof(struct testThread)*nThreads);

  for (int t=nThreads-1; t>=0; t--) {
    threads[t].args.minbytes=minBytes;
    threads[t].args.maxbytes=maxBytes;
    threads[t].args.stepbytes=stepBytes;
    threads[t].args.stepfactor=stepFactor;
    threads[t].args.localRank = localRank;

    threads[t].args.totalProcs=totalProcs;
    threads[t].args.nProcs=pcclProcs;
    threads[t].args.proc=pcclProc;
    threads[t].args.nThreads=nThreads;
    threads[t].args.thread=t;
    threads[t].args.nGpus=nGpus;
    threads[t].args.gpus=gpus+t*nGpus;
    threads[t].args.sendbuffs = sendbuffs+t*nGpus;
    threads[t].args.recvbuffs = recvbuffs+t*nGpus;
    threads[t].args.expected = expected+t*nGpus;
    threads[t].args.pcclIds = pcclIds;
    threads[t].args.comms=comms+t*nGpus;
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
    threads[t].args.devComms=devComms+t*nGpus;
#endif
    threads[t].args.streams=streams+t*nGpus;
    threads[t].args.sendRegHandles = sendRegHandles ? sendRegHandles+t*nGpus : NULL;
    threads[t].args.recvRegHandles = recvRegHandles ? recvRegHandles+t*nGpus : NULL;

    threads[t].args.barrier = (volatile int*)barrier;
    threads[t].args.barrier_idx = 0;
    threads[t].args.errors=errors+t;
    threads[t].args.bw=bw+t;
    threads[t].args.bw_count=bw_count+t;

    threads[t].args.reportErrors = datacheck;
    threads[t].args.blocking = init_blocking;

    threads[t].func = init_type == INIT_PARALLEL ? threadInit : threadRunTests;
    if (t)
      TESTCHECK(threadLaunch(threads+t));
    else
      TESTCHECK(threads[t].func(&threads[t].args));
  }

  // Wait for other threads and accumulate stats and errors
  for (int t=nThreads-1; t>=0; t--) {
    if (t) pthread_join(threads[t].thread, NULL);
    TESTCHECK(threads[t].ret);
    if (t) {
      errors[0] += errors[t];
      bw[0] += bw[t];
      bw_count[0] += bw_count[t];
    }
  }

#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (init_type != INIT_PARALLEL) {
    for(int i=0; i<nGpus*nThreads; ++i) {
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
      if (deviceImpl > 0) {
        PCCLCHECK((pcclResult_t)pcclDevCommDestroy(comms[i], devComms+i));
      }
#endif
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,0,0)
      if (local_register == SYMMETRIC_REGISTER) {
        PCCLCHECK(pcclCommWindowDeregister(comms[i], (pcclWindow_t)sendRegHandles[i]));
        PCCLCHECK(pcclCommWindowDeregister(comms[i], (pcclWindow_t)recvRegHandles[i]));
      } else
#endif
      {
        if (local_register) PCCLCHECK(pcclCommDeregister(comms[i], sendRegHandles[i]));
        if (local_register) PCCLCHECK(pcclCommDeregister(comms[i], recvRegHandles[i]));
      }
      PCCLCHECK(pcclCommDestroy(comms[i]));
    }
    free(comms);
  }

  // Free off HGGC allocated memory
  for (int i=0; i<nGpus*nThreads; i++) {
    HGGCCHECK(hggcSetDevice(gpus[i]));
    if (sendbuffs_origin[i]) PCCLCHECK(pcclMemFree((char*)sendbuffs_origin[i]));
    if (recvbuffs_origin[i]) PCCLCHECK(pcclMemFree((char*)recvbuffs_origin[i]));
    if (datacheck) PCCLCHECK(pcclMemFree(expected[i]));
  }

  free(sendRegHandles);
  free(recvRegHandles);
#if PCCL_VERSION_CODE >= PCCL_VERSION(2,2,0)
  free(devComms);
#endif

  char* envstr = getenv("NCCL_TESTS_MIN_BW");
  double check_avg_bw = envstr ? atof(envstr) : -1;
  bw[0] /= bw_count[0];

  PRINT("# Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
  PRINT("# Avg bus bandwidth    : %g %s\n", bw[0], check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw*(0.9) ? "FAILED" : "OK"));
  PRINT("#\n");
#ifdef MPI_SUPPORT
  MPI_Comm_free(&mpi_comm);
  MPI_Finalize();
#else
  barrierDeinit(proc, totalProcs);
#endif

  if (barrier) free(barrier);
#ifndef DISABLE_HGPTI
  if (elapsed_type == 2) {
    HGPTICHECK(hgptiFree());
  }
#endif

  // 'hggc-memcheck --leak-check full' requires this
  hggcDeviceReset();

  if (errors[0] || bw[0] < check_avg_bw*(0.9))
    exit(EXIT_FAILURE);
  else
    exit(EXIT_SUCCESS);
}
