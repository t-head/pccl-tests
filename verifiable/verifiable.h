#ifndef _d41d8cd98f00b204e9800998ecf8427e
#define _d41d8cd98f00b204e9800998ecf8427e

#include <hggc_runtime.h>
#include <stdint.h>

/* Routines for launching kernels that verify reduction results. A significant
 * feature of these routines is they carefully craft floating point input
 * to produce exactly predictable output.
 *
 * int elt_ty: actually just a pcclDataType_t
 *
 * int red_op: mostly just a  pcclRedOp_t. Since PreMulSum ops are dynamically
 * created, these are encoded as the value pcclNumOps and their scalar is
 * assumed to be `pcclVerifiablePremulScalar(rank_me)`
 *
 * uint64_t seed: arbitrary 64-bits to use in seeding the random values
 *
 * intptr_t elt_ix0: index of first element pointed to by elts when generating
 * random values. This makes it possible to generate subsequences independently
 * as well as in aggregate.
 *
 * int rank_n: Number of contributions into the reduction. Non-reduction
 * collectives like broadcast, gather, etc will always set this to one.
 *
 * int rank_me: Index of this contribution
 */

// Use this as the local scalar for PreMulSum ops
template<typename T>
__host__ __device__ T pcclVerifiablePremulScalar(int rank_me) {
  return T(rank_me % 2 == 0 ? 1.0f : 2.0f);
}

// Enqueue kernel to generate data which is to be reduced.
hggcError_t pcclVerifiablePrepareInput(
  void *elts, intptr_t elt_n, int elt_ty, int red_op, int rank_n, int rank_me,
  uint64_t seed, intptr_t elt_ix0, hggcStream_t stream
);

// Get generated data which is to be reduced
hggcError_t pcclVerifiableGetInput(
  void *data, int elt_type, int red_op, int rank_n, int rank_me,
  uint64_t seed, intptr_t elt_idx
);

// Enqueue kernel to generate expected results of reduction.
hggcError_t pcclVerifiablePrepareExpected(
  void *elts, intptr_t elt_n, int elt_ty, int red_op, int rank_n,
  uint64_t seed, intptr_t elt_ix0, hggcStream_t stream
);

// Enqueue kernel to verify reduced data matches expectation. The number of
// failed elements is written to bad_elt_n which must be in hggcHost memory.
// If `expected == nullptr` then the expected results are generated on-the-fly
// which can be costly. Thus if you plan to run the same reduction multiple
// times it is advantageous to precompute the expected values with
// pcclVerifiablePrepareExpected and pass them as `expected` here.
hggcError_t pcclVerifiableVerify(
  void const *results, void const *expected, intptr_t elt_n, int elt_ty,
  int red_op, int rank_n, uint64_t seed, intptr_t elt_ix0,
  int64_t *bad_elt_n, hggcStream_t stream
);

#endif
