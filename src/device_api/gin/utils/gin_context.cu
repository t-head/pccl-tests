/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "gin_context.h"
#include "common.h"

#include <hggc.h>

#define HGCHECK(cmd) do { \
  HGresult _e = (cmd); \
  if (_e != HGGC_SUCCESS) { \
    const char* _s = nullptr; \
    hgGetErrorString(_e, &_s); \
    fprintf(stderr, "Failed: HGGC driver error %s:%d %s returned %d '%s'\n", __FILE__, __LINE__, #cmd, (int)_e, _s ? _s : "unknown"); \
    return pcclUnhandledHggcError; \
  } \
} while (0)

static pcclResult_t pcclTestGinHostAlloc(void** ptr, size_t size) {
  size_t granularity = 0;
  HGdevice currentDev;
  HGmemAllocationProp prop = {};
  HGmemAccessDesc accessDesc = {};
  HGmemGenericAllocationHandle handle;
  int hggcDev, cpuNumaNodeId = -1;

  if (hggcGetDevice(&hggcDev) != hggcSuccess) return pcclUnhandledHggcError;
  HGCHECK(hgDeviceGet(&currentDev, hggcDev));
  HGCHECK(hgDeviceGetAttribute(&cpuNumaNodeId, HG_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));

  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;
  prop.location.type = HG_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = cpuNumaNodeId;
  prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  HGCHECK(hgMemGetAllocationGranularity(&granularity, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));

  size = ((size + granularity - 1) / granularity) * granularity;
  HGCHECK(hgMemCreate(&handle, size, &prop, 0));
  HGmemAllocationProp actual = {};
  HGCHECK(hgMemGetAllocationPropertiesFromHandle(&actual, handle));
  HGCHECK(hgMemAddressReserve((HGdeviceptr*)ptr, size, granularity, 0, 0));
  HGCHECK(hgMemMap((HGdeviceptr)*ptr, size, 0, handle, 0));

  accessDesc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = hggcDev;
  accessDesc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
  HGCHECK(hgMemSetAccess((HGdeviceptr)*ptr, size, &accessDesc, 1));

  accessDesc.location.type = HG_MEM_LOCATION_TYPE_HOST_NUMA;
  accessDesc.location.id = cpuNumaNodeId;
  HGCHECK(hgMemSetAccess((HGdeviceptr)*ptr, size, &accessDesc, 1));

  hggcPointerAttributes attrs = {};
  hggcError_t attrResult = hggcPointerGetAttributes(&attrs, *ptr);
  if (actual.location.type != HG_MEM_LOCATION_TYPE_HOST_NUMA &&
      (attrResult != hggcSuccess || attrs.type != hggcMemoryTypeHost)) {
    fprintf(stderr, "Error: requested HOST_NUMA allocation is not verified as host memory: "
                    "allocation location=%d runtime type=%d runtime result=%d.\n",
            (int)actual.location.type, (int)attrs.type, (int)attrResult);
    HGCHECK(hgMemUnmap((HGdeviceptr)*ptr, size));
    HGCHECK(hgMemAddressFree((HGdeviceptr)*ptr, size));
    HGCHECK(hgMemRelease(handle));
    *ptr = nullptr;
    return pcclInvalidUsage;
  }
  fprintf(stderr, "HOST allocation verified: allocation location=%d runtime type=%d runtime result=%d.\n",
          (int)actual.location.type, (int)attrs.type, (int)attrResult);
  return pcclSuccess;
}

static pcclResult_t pcclTestGinHostFree(void* ptr) {
  if (ptr == nullptr) return pcclSuccess;

  HGmemGenericAllocationHandle handle;
  size_t size = 0;

  HGCHECK(hgMemRetainAllocationHandle(&handle, ptr));
  HGCHECK(hgMemRelease(handle));
  HGCHECK(hgMemGetAddressRange(nullptr, &size, (HGdeviceptr)ptr));
  HGCHECK(hgMemUnmap((HGdeviceptr)ptr, size));
  HGCHECK(hgMemRelease(handle));
  HGCHECK(hgMemAddressFree((HGdeviceptr)ptr, size));
  return pcclSuccess;
}

/*
 * The build only produces these benchmarks against headers new enough for the
 * GIN requirements below, so this guards the remaining case the build system
 * cannot see: a library older than those headers loaded at run time.
 */
static int checkPcclVersion(void) {
  int runtimeVersion = 0;
  PCCLCHECK_FATAL(pcclGetVersion(&runtimeVersion));
  if (runtimeVersion < PCCL_VERSION(2, 2, 0)) {
    fprintf(stderr,
            "Incompatible PCCL versions. pccl-tests was compiled with PCCL %d, but is "
            "running with PCCL %d. The GIN device API is not compatible with versions "
            "before 2.2.0.\n",
            PCCL_VERSION_CODE, runtimeVersion);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return runtimeVersion;
}

size_t pcclTestGinMaxBufferBytes(void) {
  return 20ULL * (1ULL << 31);
}

void pcclTestGinDevCommCreate(pcclComm_t comm, const ginArgs_t* args, ginContext_t* ctx) {
  (void)checkPcclVersion();
  const int windowFlags = PCCL_WIN_DEFAULT;

  pcclDevCommRequirements_t reqs = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginContextCount = args->numCtas;
  reqs.ginQueueDepth = args->queueDepth;
  reqs.ginConnectionType = PCCL_GIN_CONNECTION_FULL;
  reqs.ginSignalCount = 1;
  reqs.ginCounterCount = 1;
  PCCLCHECK_FATAL(pcclDevCommCreate(comm, &reqs, &ctx->dcomm));

  if (args->localMemoryType == pcclGinMemoryHost || args->remoteMemoryType == pcclGinMemoryHost) {
    PCCLCHECK_FATAL(pcclTestGinHostAlloc(&ctx->hostBuf, args->maxBytes));
    PCCLCHECK_FATAL(pcclCommWindowRegister(comm, ctx->hostBuf, args->maxBytes, &ctx->hostBufWindow, windowFlags));
  }

  if (args->localMemoryType == pcclGinMemoryDevice || args->remoteMemoryType == pcclGinMemoryDevice) {
    PCCLCHECK_FATAL(pcclMemAlloc(&ctx->devBuf, args->maxBytes));
    PCCLCHECK_FATAL(pcclCommWindowRegister(comm, ctx->devBuf, args->maxBytes, &ctx->devBufWindow, windowFlags));
  }
}

void pcclTestGinDevCommDestroy(pcclComm_t comm, ginContext_t* ctx) {
  PCCLCHECK_FATAL(pcclDevCommDestroy(comm, &ctx->dcomm));

  if (ctx->devBufWindow != nullptr) PCCLCHECK_FATAL(pcclCommWindowDeregister(comm, ctx->devBufWindow));
  if (ctx->devBuf != nullptr) PCCLCHECK_FATAL(pcclMemFree(ctx->devBuf));

  if (ctx->hostBufWindow != nullptr) PCCLCHECK_FATAL(pcclCommWindowDeregister(comm, ctx->hostBufWindow));
  if (ctx->hostBuf != nullptr) PCCLCHECK_FATAL(pcclTestGinHostFree(ctx->hostBuf));
}
