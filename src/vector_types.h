/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _VECTOR_TYPES_H_
#define _VECTOR_TYPES_H_

#include "hggc_runtime.h"

template <typename T>
struct VectorTypeMapping{
  using Type=T;
};

template <>
struct VectorTypeMapping<float>{
  using Type=float4;
};

template <>
struct VectorTypeMapping<double>{
  using Type=double2;
};

template <>
struct VectorTypeMapping<int8_t>{
  using Type=char4;
};

template <>
struct VectorTypeMapping<uint8_t>{
  using Type=uchar4;
};

template <>
struct VectorTypeMapping<int32_t>{
  using Type=int4;
};

template <>
struct VectorTypeMapping<uint32_t>{
  using Type=uint4;
};

template <typename T>
__device__ __forceinline__ T vectorAdd(T a, T b) {
  return a + b;
}

template <>
__device__ __forceinline__ float4 vectorAdd(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <>
__device__ __forceinline__ double2 vectorAdd(double2 a, double2 b) {
  return make_double2(a.x + b.x, a.y + b.y);
}

template <>
__device__ __forceinline__ char4 vectorAdd(char4 a, char4 b) {
  return make_char4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <>
__device__ __forceinline__ uchar4 vectorAdd(uchar4 a, uchar4 b) {
  return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <>
__device__ __forceinline__ int4 vectorAdd(int4 a, int4 b) {
  return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <>
__device__ __forceinline__ uint4 vectorAdd(uint4 a, uint4 b) {
  return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

#endif // _VECTOR_TYPES_H_
