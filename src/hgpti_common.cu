/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "hgpti_common.h"

#define BUF_SIZE (16 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

int *kernelDevIdx = NULL;
uint64_t *kernelDurations = NULL;
uint8_t *activityRecordkBuffer = NULL;
static int maxRecordCountPerRank = 0;
static int maxDevCount = 0;
static int kernelCount = 1;
static bool clean = true;
static bool hgptiError = false;

void hgptiBuffersCallbackRequestFunc(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  *size = BUF_SIZE * maxRecordCountPerRank;
  *buffer = ALIGN_BUFFER(activityRecordkBuffer, ALIGN_SIZE);
  *maxNumRecords = maxRecordCountPerRank * maxDevCount * kernelCount;
}

void hgptiBuffersCallbackCompleteFunc(HGcontext context, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
  if (clean) return;

  HGpti_Activity *record = NULL;
  HGptiResult status = HGPTI_SUCCESS;
  do {
    status = hgptiActivityGetNextRecord(buffer, validSize, &record);
    if(status == HGPTI_SUCCESS) {
      const HGpti_ActivityKernel6 *kernel = (HGpti_ActivityKernel6*)record;
      if (kernel->kind != HGPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        printf("error kernel kind: %d ?\n", kernel->kind);
        goto fail;
      }
      uint32_t deviceId = kernel->deviceId;
      uint64_t start = kernel->start;
      uint64_t end = kernel->end;

      // printf("kernel name: %s, dev: %d, kernel id :%d\n", kernel->name, deviceId, kernelDevIdx[deviceId]);
      if (start == 0 && end == 0) {
        printf("timestamp information could not be collected for the kernel.\n");
        goto fail;
      } else {
        if (kernelDevIdx[deviceId] >= maxRecordCountPerRank * kernelCount) {
          printf("too more record count in dev:%d, now is %d, max number is %d\n", deviceId, kernelDevIdx[deviceId], maxRecordCountPerRank);
          goto fail;
        }
        uint64_t kernelTime = end - start;
        kernelDurations[maxRecordCountPerRank * kernelCount * deviceId + kernelDevIdx[deviceId]] = kernelTime;
        kernelDevIdx[deviceId]++;
      }
    } else if (status == HGPTI_ERROR_MAX_LIMIT_REACHED) {
      for (int i = 0; i < maxDevCount; i++) {
        if (kernelDevIdx[i] != 0 && kernelDevIdx[i] != maxRecordCountPerRank * kernelCount) {
          printf("kernelDevIdx in dev:%d is %d ?\n", i, kernelDevIdx[i]);
          goto fail;
        }
      }
      break;
    } else {
      printf("hgpti callback error, status is %d\n", status);
      goto fail;
    }
  } while (1);

exit:
  return;
fail:
  hgptiError = true;
  goto exit;
}

testResult_t hgptiInit(int iters, int kerNum, int devNum) {
  maxRecordCountPerRank = iters;
  maxDevCount = devNum;
  kernelCount = kerNum;
  hgptiError = false;

  kernelDevIdx = (int *)calloc(maxDevCount, sizeof(int));
  kernelDurations = (uint64_t *)calloc(maxRecordCountPerRank * maxDevCount * kernelCount, sizeof(uint64_t));
  activityRecordkBuffer = (uint8_t *)malloc(BUF_SIZE * maxRecordCountPerRank * maxDevCount * kernelCount + ALIGN_SIZE);
  if (!kernelDurations || !activityRecordkBuffer) {
    printf("HGPTI buffers callback error: out of memory.\n");
    return testHgptiError;
  }

  HGPTICHECK(hgptiActivityEnable(HGPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  HGPTICHECK(hgptiActivityRegisterCallbacks(hgptiBuffersCallbackRequestFunc, hgptiBuffersCallbackCompleteFunc));

  return testSuccess;
}

testResult_t hgptiFree() {
  if (kernelDevIdx) free(kernelDevIdx);
  if (kernelDurations) free(kernelDurations);
  if (activityRecordkBuffer) free(activityRecordkBuffer);
  HGPTICHECK(hgptiFinalize());

  return testSuccess;
}

testResult_t hgptiClean() {
  hggcDeviceSynchronize();
  clean = true;
  hgptiError = false;

  HGPTICHECK(hgptiActivityFlushAll(HGPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  HGPTICHECK(hgptiActivityEnable(HGPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  memset(kernelDevIdx, 0, maxDevCount * sizeof(int));
  memset(kernelDurations, 0, maxRecordCountPerRank * maxDevCount * kernelCount * sizeof(uint64_t));
  memset(activityRecordkBuffer, 0, BUF_SIZE * maxRecordCountPerRank * maxDevCount * kernelCount + ALIGN_SIZE);

  return testSuccess;
}

testResult_t hgptiFlush(double *deltaSecPerGpusPerIter) {
  hggcDeviceSynchronize();
  clean = false;
  hgptiError = false;

  HGPTICHECK(hgptiActivityFlushAll(HGPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  HGPTICHECK(hgptiActivityDisable(HGPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  if (hgptiError) return testHgptiError;

  int devId = 0;
  for (int i = 0; i < maxDevCount; i++) {
    if (kernelDevIdx[i] == 0) continue;

    for (int j = 0; j < maxRecordCountPerRank; j++) {
      for (int k = 0; k < kernelCount; k++) {
        deltaSecPerGpusPerIter[devId * maxRecordCountPerRank + j] +=
            (double)kernelDurations[maxRecordCountPerRank * kernelCount * i + kernelCount * j + k];
      }
      deltaSecPerGpusPerIter[devId * maxRecordCountPerRank + j] /= 1e3;
    }
    devId++;
  }

  return testSuccess;
}