/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Communicator lifecycle benchmark derived from nccl-tests v2.22.0.

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <algorithm>
#include <vector>

#ifndef MPI_SUPPORT
int main() {
  printf("This test requires MPI support, but was compiled without it.\n");
  return 1;
}
#else

#include "pccl.h"
#include "mpi.h"
#include "../verifiable/verifiable.h"

#if PCCL_VERSION_CODE >= PCCL_VERSION(2,3,0)

#define ABORT_EXIT do { \
  char hostname[1024]; \
  gethostname(hostname, sizeof(hostname)); \
  hostname[sizeof(hostname)-1] = '\0'; \
  fprintf(stderr, "%s: Aborting due to error in %s:%d\n", hostname, __FILE__, __LINE__); \
  fflush(stderr); fflush(stdout); \
  int initialized; \
  MPI_Initialized(&initialized); \
  if (initialized) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
  exit(EXIT_FAILURE); \
} while(0)

#define HGGCCHECK(cmd) do {                         \
  hggcError_t e = cmd;                              \
  if( e != hggcSuccess ) {                          \
    printf("HGGC failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,hggcGetErrorString(e));   \
    ABORT_EXIT;                                     \
  }                                                 \
} while(0)

#define PCCLCHECK(cmd) do {                         \
  pcclResult_t r = cmd;                             \
  if (r != pcclSuccess) {                           \
    printf("PCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,pcclGetErrorString(r));   \
    ABORT_EXIT;                                     \
  }                                                 \
} while(0)

#define MPICHECK(cmd) do {                         \
  int r = cmd;                                     \
  if (r != MPI_SUCCESS) {                          \
    int err_len;                                   \
    char err_string[MPI_MAX_ERROR_STRING];         \
    MPI_Error_string(r, err_string, &err_len);     \
    printf("MPI failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,err_string); \
    ABORT_EXIT;                                     \
  }                                                 \
} while(0)

int main_ret_val = 0;

void print_help();

enum timer_enums {
  mpi_init,
  mpi_barrier,
  mpi_allgatherv,
  work_prepare,
  pccl_init,
  pccl_split,
  pccl_grow,
  pccl_allreduce,
  pccl_finalize,
  pccl_destroy,
  pccl_abort,
  pccl_shrink,
  hggc_init,
  hggc_stream_sync,
  timer_enum_count,
};

// timer_strings array - order must match timer_enums exactly
const char* const timer_strings[timer_enum_count] = {
  "(MPI) Initialization",           // mpi_init
  "(MPI) Barrier",                  // mpi_barrier
  "(MPI) Allgatherv for pcclUniqueId", // mpi_allgatherv
  "(HGGC) Allocate mem and streams",   // work_prepare
  "PCCL pcclCommInitRankScalable",  // pccl_init
  "PCCL pcclCommSplit",             // pccl_split
  "PCCL pcclCommGrow",              // pccl_grow
  "PCCL pcclAllReduce (host-call)", // pccl_allreduce
  "PCCL pcclCommFinalize",          // pccl_finalize
  "PCCL pcclCommDestroy",           // pccl_destroy
  "PCCL pcclCommAbort",             // pccl_abort
  "PCCL pcclCommShrink",            // pccl_shrink
  "(HGGC) Initialization",          // hggc_init
  "(HGGC) Stream Synchronize",      // hggc_stream_sync
};

const int pccl_flag = 0x1 << 1;
const int hggc_flag = 0x1 << 2;
const int mpi_flag  = 0x1 << 3;
// timer_flags array - order must match timer_enums exactly
const int timer_flags[timer_enum_count] = {
  mpi_flag,   // mpi_init
  mpi_flag,   // mpi_barrier
  mpi_flag,   // mpi_allgatherv
  hggc_flag,  // work_prepare
  pccl_flag,  // pccl_init
  pccl_flag,  // pccl_split
  pccl_flag,  // pccl_grow
  pccl_flag,  // pccl_allreduce
  pccl_flag,  // pccl_finalize
  pccl_flag,  // pccl_destroy
  pccl_flag,  // pccl_abort
  pccl_flag,  // pccl_shrink
  hggc_flag,  // hggc_init
  hggc_flag,  // hggc_stream_sync
};
bool timer_type_pccl(int timer_enum) {
  return (timer_flags[timer_enum] & pccl_flag) == pccl_flag;
}

struct timer_record_t {
  int flush_interval;
  bool currently_enabled = true;
  int count = 0;
  MPI_Comm comm = MPI_COMM_NULL;
  int rank = 0;
  int rank_count = 1;
  const char* role;
  int sweep_size;
  int color;

  uint64_t timer_enum_start[timer_enum_count];

  /* locally-written for each timing call. */
  double* local;
  double* local_sq;
  int* id;
  int* participated;

  /* root only, reduced from all ranks. */
  double* max;
  double* sum;
  double* sumsq;
  int* participant_count;

  /* root-only, sorted by timer type. */
  int rounds_by_type[timer_enum_count];
  double sum_by_type[timer_enum_count];
  double sumsq_by_type[timer_enum_count];
  double max_by_type[timer_enum_count];
  int participant_count_by_type[timer_enum_count];
};

void flush_timing(timer_record_t* timing);
timer_record_t* timing_allocate(MPI_Comm comm, int flush, const char* role="initial", int sweep_size=0, int color=0);

inline uint64_t clockNano() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec)*1000*1000*1000 + ts.tv_nsec;
}

enum test_kinds {
  test_init,
  test_split,
  test_shrink,
  test_grow
};
enum step_kinds {
  kind_factor,
  kind_step,
  kind_fixed,
};

struct args_t {
  int timing_flush_interval;
  int warmup;
  int resize_kind;
  double resize_num;
  int iterations;
  int work_iterations;
  int test_kind;
  bool aborts;
  int beginranks;
  int endranks;
  int sweep_kind;
  double sweep_num;
  int verbose;
  int pccl_only;
  int share;
  int root_every;
  int check;
} args;

int parse_integer(const char* arg, int minimum) {
  char* end;
  errno = 0;
  long value = strtol(arg, &end, 10);
  if (errno || end == arg || *end || value < minimum || value > INT_MAX) {
    fprintf(stderr, "Error: Expected an integer in [%d, %d], got '%s'\n", minimum, INT_MAX, arg);
    ABORT_EXIT;
  }
  return (int)value;
}

// Steps are positive; factors greater than one specify the resize ratio.
int parse_step_kinds(char *arg, int *kind, double *num) {
  char *numpart;
  if (strncmp(arg, "factor-", 7) == 0) {
    *kind = kind_factor;
    numpart = &arg[7];
  } else if (strncmp(arg, "step-",5) == 0) {
    *kind = kind_step;
    numpart = &arg[5];
  } else if (strcmp(arg, "fixed") == 0) {
    *kind = kind_fixed;
    *num = 0.0;
    return 0;
  } else {
    fprintf(stderr, "Error: Unknown sweep or resize kind '%s'.  Expected factor-N.NN, step-N, or fixed.\n", arg);
    *num = 0.0;
    return 1;
  }
  errno = 0;
  char* end;
  *num = strtod(numpart, &end);
  if (errno || end == numpart || *end || !isfinite(*num) || *num <= 0.0 || *num > INT_MAX) {
    fprintf(stderr, "Error: invalid number (\"%s\").  Expected positive finite number.\n", numpart);
    return 1;
  }

  if (*kind == kind_factor && *num < 1.0) {
    *num = 1.0/(*num);
  }
  if ((*kind == kind_factor && (*num == 1.0 || *num > INT_MAX)) ||
      (*kind == kind_step && floor(*num) != *num)) {
    fprintf(stderr, "Error: Expected a factor other than 1 or a positive integer step\n");
    return 1;
  }
  return 0;
}

void parse_args(int argc, char **argv) {
  args.resize_kind = kind_factor;
  args.resize_num = 2.0;
  args.test_kind = test_init;
  args.aborts = false;
  args.verbose = 0;
  args.timing_flush_interval = 1;
  args.sweep_kind = kind_factor;
  args.sweep_num  = 2;
  args.beginranks = 0;
  args.endranks = 0;
  args.warmup = 1;
  args.pccl_only = 1;
  args.work_iterations = 2;
  args.iterations = 2;
  args.root_every = 0;
  args.check = 1;

  if (argc < 2) {
    fprintf(stderr, "Error: You must specify the test to execute: init, split, shrink, or grow\n");
    print_help();
    ABORT_EXIT;
  }
  // Reset getopt state and skip the test type argument
  optind = 1;
  if (strcmp(argv[1], "init") == 0) {
    args.test_kind = test_init;
    optind++;
  } else if (strcmp(argv[1], "split") == 0) {
    args.test_kind = test_split;
    optind++;
  } else if (strcmp(argv[1], "shrink") == 0) {
    args.test_kind = test_shrink;
    optind++;
  } else if (strcmp(argv[1], "grow") == 0) {
    args.test_kind = test_grow;
    optind++;
  } else if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "help") == 0) {
    print_help();
    MPICHECK(MPI_Finalize());
    exit(0);
  } else {
    fprintf(stderr, "Error: Unknown test type '%s'. Must be one of: init, split, shrink, or grow\n", argv[1]);
    print_help();
    ABORT_EXIT;
  }

  static struct option long_options[] = {
    {"abort",          no_argument,       0, 'a'},
    {"all-timers",     no_argument,       0, 'A'},
    {"sweep-begin",    required_argument, 0, 'b'},
    {"check",          required_argument, 0, 'c'},
    {"sweep-end",      required_argument, 0, 'e'},
    {"flush-interval", required_argument, 0, 'F'},
    {"help",           no_argument,       0, 'h'},
    {"iterations",     required_argument, 0, 'i'},
    {"share",          no_argument,       0, 'S'},
    {"resize",         required_argument, 0, 'r'},
    {"root-every",     required_argument, 0, 'R'},
    {"sweep-comms",    required_argument, 0, 's'},
    {"verbose",        no_argument,       0, 'v'},
    {"warmup",         required_argument, 0, 'w'},
    {"work-iterations",required_argument, 0, 'W'},
    {0, 0, 0, 0}
  };

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "aAb:c:e:F:hi:r:R:s:Svw:W:", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'a':
        args.aborts = true;
        break;
      case 'b':
        args.beginranks = parse_integer(optarg, 1);
        break;
      case 'c':
        args.check = parse_integer(optarg, 0);
        if (args.check > 1) {
          fprintf(stderr, "Error: Check must be 0 or 1\n");
          ABORT_EXIT;
        }
        break;
      case 'e':
        args.endranks = parse_integer(optarg, 1);
        break;
      case 's':
        if (parse_step_kinds(optarg, &args.sweep_kind, &args.sweep_num)) {
          print_help();
          ABORT_EXIT;
        }
        break;
      case 'F':
        args.timing_flush_interval = parse_integer(optarg, 1);
        break;
      case 'h':
        print_help();
        MPICHECK(MPI_Finalize());
        exit(0);
        break;
      case 'i':
        args.iterations = parse_integer(optarg, 1);
        break;
      case 'r':
        if (parse_step_kinds(optarg, &args.resize_kind, &args.resize_num)) {
          print_help();
          ABORT_EXIT;
        }
        break;
      case 'A':
        args.pccl_only = 0;
        break;
      case 'R':
        args.root_every = parse_integer(optarg, 0);
        break;
      case 'S':
        args.share = true;
        break;
      case 'v':
        args.verbose = 1;
        break;
      case 'w':
        args.warmup = parse_integer(optarg, 0);
        break;
      case 'W':
        args.work_iterations = parse_integer(optarg, 1);
        break;
      default:
        print_help();
        fprintf(stderr, "Error: Unknown option '%s'\n", argv[optind-1]);
        ABORT_EXIT;
    }
  }

  int nranks;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  if (optind != argc || args.resize_kind == kind_fixed || args.iterations > INT_MAX - args.warmup) {
    fprintf(stderr, "Error: Invalid arguments\n");
    ABORT_EXIT;
  }

  if (args.beginranks == 0) {
    args.beginranks = nranks;
  }
  if (args.endranks == 0) {
    args.endranks = 1;
  }

  if (args.beginranks > nranks || args.endranks > nranks) {
    fprintf(stderr, "Error: Sweep rank counts must not exceed MPI size %d\n", nranks);
    ABORT_EXIT;
  }

  bool going_up = args.beginranks < args.endranks;
  if (args.sweep_kind == kind_factor) {
    bool _going_up = args.sweep_num > 1.0;
    if (going_up != _going_up)
      args.sweep_num = 1.0 / args.sweep_num;
  } else if (args.sweep_kind == kind_step) {
    bool _going_up = args.sweep_num > 0;
    if (going_up != _going_up)
      args.sweep_num = -args.sweep_num;
  }

  if (args.test_kind == test_split && args.share && args.aborts) {
    fprintf(stderr, "Error: split test with both --share and --abort is not valid.\n");
    ABORT_EXIT;
  }

}

static inline void record_timer(timer_record_t* timing, uint64_t elapsed, int timer_enum, bool record) {
  if (!timing->currently_enabled) return;
  assert(timer_enum >= 0 && timer_enum < timer_enum_count);
  if (timing->count < timing->flush_interval) {
    timing->local[timing->count] = (double)elapsed;
    timing->id[timing->count] = timer_enum;
    timing->participated[timing->count] = record;
    timing->local_sq[timing->count] = pow(timing->local[timing->count], 2.0);
  }
  timing->count++;
  if (timing->count == timing->flush_interval) {
    flush_timing(timing);
  }
}

static inline void set_timer_enabled(timer_record_t* timing, bool enabled) {
  timing->currently_enabled = enabled;
}

static inline void start_timer(timer_record_t* timing, int timer_enum) {
  if (!timing->currently_enabled) return;
  timing->timer_enum_start[timer_enum] = clockNano();
}

static inline void stop_timer(timer_record_t* timing, int timer_enum, bool record=true) {
  if (!timing->currently_enabled) return;
  uint64_t elapsed;
  if (record) {
    elapsed = clockNano() - timing->timer_enum_start[timer_enum];
  } else {
    elapsed = 0;
  }
  record_timer(timing, elapsed, timer_enum, record);
}

inline void skip_timer(timer_record_t* timing, int timer_enum) {
  record_timer(timing, 0, timer_enum, false);
}

struct work_t {
  int *ddata;
  int *ddata_out;
  int64_t *wrong;
  int64_t total_wrong;
  uint64_t total_checked;
  uint64_t seed;
  hggcStream_t stream;
} work_ctx;

const int worksize = 1024;

void prepare_work(timer_record_t* timing) {
  start_timer(timing, work_prepare);
  HGGCCHECK(hggcStreamCreateWithFlags(&work_ctx.stream, hggcStreamNonBlocking));
  HGGCCHECK(hggcMalloc(&work_ctx.ddata, worksize*sizeof(int)));
  HGGCCHECK(hggcMalloc(&work_ctx.ddata_out, worksize*sizeof(int)));
  HGGCCHECK(hggcHostAlloc((void**)&work_ctx.wrong, sizeof(int64_t), hggcHostAllocMapped));
  stop_timer(timing, work_prepare);
}

void do_work(timer_record_t* timing, pcclComm_t comm) {
  uint64_t seed = 2*work_ctx.seed++;
  if (timing == NULL) return;
  // Split groups use distinct seeds within each iteration.
  seed += timing->color;
  int world_rank;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  const char* tests[] = {"init", "split", "shrink", "grow"};
  if (comm == NULL) {
    fprintf(stderr, "Error: Missing %s communicator: world rank %d, role %s, color %d, sweep size %d\n",
      tests[args.test_kind], world_rank, timing->role, timing->color, timing->sweep_size);
    ABORT_EXIT;
  }
  int rank, nranks;
  PCCLCHECK(pcclCommUserRank(comm, &rank));
  PCCLCHECK(pcclCommCount(comm, &nranks));
  if (rank != timing->rank || nranks != timing->rank_count) {
    fprintf(stderr, "Error: Unexpected %s communicator: world rank %d, comm rank %d/%d, expected %d/%d, role %s, color %d, sweep size %d\n",
      tests[args.test_kind], world_rank, rank, nranks, timing->rank, timing->rank_count,
      timing->role, timing->color, timing->sweep_size);
    ABORT_EXIT;
  }
  HGGCCHECK(pcclVerifiablePrepareInput(work_ctx.ddata, worksize, pcclInt, pcclSum, timing->rank_count, timing->rank, seed, 0, work_ctx.stream));
  HGGCCHECK(hggcStreamSynchronize(work_ctx.stream));
  start_timer(timing, pccl_allreduce);
  PCCLCHECK(pcclAllReduce(work_ctx.ddata, work_ctx.ddata_out, worksize, pcclInt, pcclSum, comm, work_ctx.stream));
  stop_timer(timing, pccl_allreduce);
  start_timer(timing, hggc_stream_sync);
  HGGCCHECK(hggcStreamSynchronize(work_ctx.stream));
  stop_timer(timing, hggc_stream_sync);
  if (args.check) {
    HGGCCHECK(pcclVerifiableVerify(work_ctx.ddata_out, NULL, worksize, pcclInt, pcclSum, timing->rank_count, seed, 0, work_ctx.wrong, work_ctx.stream));
    HGGCCHECK(hggcStreamSynchronize(work_ctx.stream));
    if (*work_ctx.wrong) {
      fprintf(stderr, "AllReduce verification failed after %s: world rank %d, comm rank %d/%d, role %s, color %d, sweep size %d, seed %llu, datatype int32, op sum, count %d, wrong %lld\n",
        tests[args.test_kind], world_rank, rank, nranks, timing->role, timing->color,
        timing->sweep_size, (unsigned long long)seed, worksize, (long long)*work_ctx.wrong);
    }
    work_ctx.total_wrong += *work_ctx.wrong;
    work_ctx.total_checked++;
  }
}

void cleanup_work() {
  HGGCCHECK(hggcFree(work_ctx.ddata));
  HGGCCHECK(hggcFree(work_ctx.ddata_out));
  HGGCCHECK(hggcFreeHost(work_ctx.wrong));
  HGGCCHECK(hggcStreamDestroy(work_ctx.stream));
}

static int rootCount(int nranks) {
  if (args.root_every > 0) {
    return 1 + ((nranks-1) / args.root_every);
  } else {
    return 1;
  }
}

static int isRoot(int rank) {
  if (args.root_every > 0)
    return (rank % args.root_every) == 0;
  else
    return rank == 0;
}

void timing_print_verbose(timer_record_t* timing) {
  int nranks;
  if (args.verbose == 0) return;
  MPICHECK(MPI_Comm_size(timing->comm, &nranks));
  int nrec = timing->count < timing->flush_interval ? timing->count : timing->flush_interval;
  for (int i=0; i<nrec; i++) {
    if (timing->participant_count[i] == 0) continue;
    if (args.pccl_only && !timer_type_pccl(timing->id[i])) continue;
    double avg = timing->sum[i] / timing->participant_count[i];
    double A = timing->sumsq[i] / timing->participant_count[i];
    double B = pow(timing->sum[i] / timing->participant_count[i], 2.0);
    double stddev = sqrt(std::max(0.0, A - B));
    printf("%36s: Comm size %4d, Role %7s, Color %d, Sweep size %4d, Max %9.3f ms, Avg %8.3f ms, Stddev %8.3f ms, [%4d/%4d]\n", \
      timer_strings[timing->id[i]], nranks, timing->role, timing->color, timing->sweep_size,
      timing->max[i]*1e-6, avg*1e-6, stddev*1e-6, (int)timing->participant_count[i], nranks);
  }
  fflush(stdout);
}

// All ranks must record the same sequence of timer events.
void gather_timing(timer_record_t* timing) {
  int *all_counts = (int*)malloc(sizeof(int) * timing->rank_count);

  if (timing->comm == MPI_COMM_NULL) {
    all_counts[0] = timing->count;
    for (int i=0; i<timing->count; i++) {
      timing->max[i] = timing->local[i];
      timing->sum[i] = timing->local[i];
      timing->participant_count[i] = timing->participated[i];
      timing->sumsq[i] = timing->local_sq[i];
    }
  } else {
    MPICHECK(MPI_Gather(&timing->count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, timing->comm));
    MPICHECK(MPI_Reduce(timing->local, timing->max, timing->count, MPI_DOUBLE, MPI_MAX, 0, timing->comm));
    MPICHECK(MPI_Reduce(timing->local, timing->sum, timing->count, MPI_DOUBLE, MPI_SUM, 0, timing->comm));
    MPICHECK(MPI_Reduce(timing->participated, timing->participant_count, timing->count, MPI_INT, MPI_SUM, 0, timing->comm));
    MPICHECK(MPI_Reduce(timing->local_sq, timing->sumsq, timing->count, MPI_DOUBLE, MPI_SUM, 0, timing->comm));
  }

  if (timing->rank == 0) {
    for (int i=0; i<timing->rank_count; i++) {
      if (all_counts[i] != timing->count) {
        printf("INTERNAL TEST ERROR: Rank %d: timer_records_count mismatch: %d != %d\n", i, all_counts[i], timing->count);
        main_ret_val = 1;
      }
    }
    for (int jj=0; jj<timing->count; jj++) {
      int id = timing->id[jj];
      if (timing->participant_count[jj] > 0) {
        timing->sum_by_type[id] += timing->sum[jj];
        timing->sumsq_by_type[id] += timing->sumsq[jj];
        timing->max_by_type[id] = std::max(timing->max_by_type[id], timing->max[jj]);
        timing->participant_count_by_type[id] += timing->participant_count[jj];
        timing->rounds_by_type[id]++;
      }
    }
    if (timing->count > timing->flush_interval) {
      printf("ERROR: Max timer records (%d) exceeded limit (%d)\n", timing->count, timing->flush_interval);
      main_ret_val = 1;
    }
  }
  free(all_counts);
}

void flush_timing(timer_record_t* timing) {
  if (timing->comm != MPI_COMM_NULL) {
    int rank;
    gather_timing(timing);

    MPICHECK(MPI_Comm_rank(timing->comm, &rank));
    if (rank == 0) {
      timing_print_verbose(timing);
    }
    MPICHECK(MPI_Barrier(timing->comm));
  }
  timing->count = 0;
}

void configure_communicator(pcclConfig_t* config) {
  config->blocking = 1;
  config->splitShare = args.share ? 1 : 0;
  config->shrinkShare = args.share ? 1 : 0;
}

void main_init(int argc, char **argv, timer_record_t* timing, MPI_Comm mpicomm) {
  pcclUniqueId commId;
  pcclComm_t pcclcomm;
  int nranks, rank;
  int nroots;

  set_timer_enabled(timing, 0 >= args.warmup);

  MPICHECK(MPI_Comm_size(mpicomm, &nranks));
  MPICHECK(MPI_Comm_rank(mpicomm, &rank));

  start_timer(timing, mpi_barrier);
  MPICHECK(MPI_Barrier(mpicomm));
  stop_timer(timing, mpi_barrier);

  nroots = rootCount(nranks);
  pcclUniqueId *id_array= (pcclUniqueId*)calloc(nroots, PCCL_UNIQUE_ID_BYTES);

  for (int j=0; j<args.iterations + args.warmup; j++) {
    set_timer_enabled(timing, j >= args.warmup);

    if (isRoot(rank)){
      PCCLCHECK(pcclGetUniqueId(&commId));
    }

    int* recv_count = (int*)malloc(sizeof(int) * nranks);
    int* recv_displ = (int*)malloc(sizeof(int) * nranks);
    int c=0;
    for (int i=0; i<nranks; ++i){
      int peer_size = isRoot(i) ? PCCL_UNIQUE_ID_BYTES : 0;
      recv_displ[i] = c;
      recv_count[i] = peer_size;
      c+= peer_size;
    }
    start_timer(timing, mpi_allgatherv);
    MPICHECK(MPI_Allgatherv(&commId,recv_count[rank],MPI_CHAR,id_array,recv_count,recv_displ,MPI_CHAR,mpicomm));
    stop_timer(timing, mpi_allgatherv);
    free(recv_count);
    free(recv_displ);

    start_timer(timing, mpi_barrier);
    MPICHECK(MPI_Barrier(mpicomm));
    stop_timer(timing, mpi_barrier);

    pcclConfig_t config = PCCL_CONFIG_INITIALIZER;
    configure_communicator(&config);
    start_timer(timing, pccl_init);
    PCCLCHECK(pcclCommInitRankScalable(&pcclcomm, nranks, rank, nroots, id_array, &config));
    stop_timer(timing, pccl_init);

    for (int i=0; i<args.work_iterations; i++) {
      do_work(timing, pcclcomm);
    }

    if (args.aborts) {
      start_timer(timing, pccl_abort);
      PCCLCHECK(pcclCommAbort(pcclcomm));
      stop_timer(timing, pccl_abort);
    } else {
      start_timer(timing, pccl_finalize);
      PCCLCHECK(pcclCommFinalize(pcclcomm));
      stop_timer(timing, pccl_finalize);
      start_timer(timing, pccl_destroy);
      PCCLCHECK(pcclCommDestroy(pcclcomm));
      stop_timer(timing, pccl_destroy);
    }

  }

  free(id_array);
}

static inline int resize_count(int count) {
  if (args.resize_kind == kind_factor) {
    return (int)ceil(count/args.resize_num);
  } else {
    return count - (int)args.resize_num;
  }
}

// Color 0 keeps the first ceil(count/factor) or count-step ranks.
static inline int pick_color(int rank, int count) {
  return rank >= resize_count(count);
}

// Each iteration derives a new child from the same parent.
void main_split(int argc, char **argv, timer_record_t* timing, MPI_Comm mpicomm, std::vector<timer_record_t*>& all_timings) {
  pcclUniqueId commId;
  pcclComm_t pcclcomm;
  int nranks, rank;

  set_timer_enabled(timing, 0 >= args.warmup);

  MPICHECK(MPI_Comm_size(mpicomm, &nranks));
  MPICHECK(MPI_Comm_rank(mpicomm, &rank));

  if (nranks == 1) return;

  start_timer(timing, mpi_barrier);
  MPICHECK(MPI_Barrier(mpicomm));
  stop_timer(timing, mpi_barrier);

  {
    int nroots = rootCount(nranks);
    pcclUniqueId *id_array= (pcclUniqueId*)calloc(nroots, PCCL_UNIQUE_ID_BYTES);
    if (isRoot(rank)){
      PCCLCHECK(pcclGetUniqueId(&commId));
    }

    int* recv_count = (int*)malloc(sizeof(int) * nranks);
    int* recv_displ = (int*)malloc(sizeof(int) * nranks);
    int c=0;
    for (int i=0; i<nranks; ++i){
      int peer_size = isRoot(i) ? PCCL_UNIQUE_ID_BYTES : 0;
      recv_displ[i] = c;
      recv_count[i] = peer_size;
      c+= peer_size;
    }
    start_timer(timing, mpi_allgatherv);
    MPICHECK(MPI_Allgatherv(&commId,recv_count[rank],MPI_CHAR,id_array,recv_count,recv_displ,MPI_CHAR,mpicomm));
    stop_timer(timing, mpi_allgatherv);

    free(recv_count);
    free(recv_displ);

    start_timer(timing, pccl_init);
    pcclConfig_t config = PCCL_CONFIG_INITIALIZER;
    configure_communicator(&config);
    PCCLCHECK(pcclCommInitRankScalable(&pcclcomm, nranks, rank, nroots, id_array, &config));
    stop_timer(timing, pccl_init);
    free(id_array);
  }

  int *exclude_ranks = (int*)malloc(sizeof(int) * nranks);

  int color = pick_color(rank, nranks);
  MPI_Comm child_comm;
  MPICHECK(MPI_Comm_split(mpicomm, args.test_kind == test_shrink && color != 0 ? MPI_UNDEFINED : color, rank, &child_comm));
  timer_record_t* child_timing = NULL;
  if (child_comm != MPI_COMM_NULL) {
    child_timing = timing_allocate(child_comm, args.timing_flush_interval, "child", nranks, color);
    all_timings.push_back(child_timing);
  }

  pcclComm_t split_child;

  for (int j=0; j<args.iterations + args.warmup; j++) {
    set_timer_enabled(timing, j >= args.warmup);
    if (child_timing != NULL) set_timer_enabled(child_timing, j >= args.warmup);

    int pcclCount;
    int exclude_count = 0;
    int shrink_flag = args.aborts ? PCCL_SHRINK_ABORT : PCCL_SHRINK_DEFAULT;

    MPICHECK(MPI_Barrier(mpicomm));

    PCCLCHECK(pcclCommCount(pcclcomm, &pcclCount));

    if (pcclCount == 1) break;

    int new_rank=0;

    for (int other_rank=0; other_rank<pcclCount; other_rank++) {
      if (color != pick_color(other_rank, pcclCount) ) {
        exclude_ranks[exclude_count++] = other_rank;
      }
    }

    split_child = NULL;
    if (args.test_kind == test_split) {
      start_timer(timing, pccl_split);
      PCCLCHECK(pcclCommSplit(pcclcomm, color, new_rank, &split_child, NULL));
      stop_timer(timing, pccl_split);
    } else if (args.test_kind == test_shrink) {
      start_timer(timing, pccl_shrink);
      if (color == 0) {
        PCCLCHECK(pcclCommShrink(pcclcomm, exclude_ranks, exclude_count, &split_child, NULL, shrink_flag));
        stop_timer(timing, pccl_shrink);
      } else {
        skip_timer(timing, pccl_shrink);
      }
    }

    for (int i=0; i<args.work_iterations; i++) {
      do_work(child_timing, split_child);
    }

    if (split_child != NULL) {
      if (args.aborts) {
        start_timer(child_timing, pccl_abort);
        PCCLCHECK(pcclCommAbort(split_child));
        stop_timer(child_timing, pccl_abort);
      } else {
        start_timer(child_timing, pccl_finalize);
        PCCLCHECK(pcclCommFinalize(split_child));
        stop_timer(child_timing, pccl_finalize);
        start_timer(child_timing, pccl_destroy);
        PCCLCHECK(pcclCommDestroy(split_child));
        stop_timer(child_timing, pccl_destroy);
      }
    }
  }

  MPICHECK(MPI_Barrier(mpicomm));
  if (args.aborts) {
    start_timer(timing, pccl_abort);
    PCCLCHECK(pcclCommAbort(pcclcomm));
    stop_timer(timing, pccl_abort);
  } else {
    start_timer(timing, pccl_finalize);
    PCCLCHECK(pcclCommFinalize(pcclcomm));
    stop_timer(timing, pccl_finalize);
    start_timer(timing, pccl_destroy);
    PCCLCHECK(pcclCommDestroy(pcclcomm));
    stop_timer(timing, pccl_destroy);
  }
  free(exclude_ranks);
}

void main_grow(int argc, char **argv, timer_record_t* timing, MPI_Comm mpicomm, std::vector<timer_record_t*>& all_timings) {
  pcclUniqueId commId_smaller, commId_bigger;
  pcclComm_t pcclComm_smaller, pcclComm_bigger;
  int nranks, rank, color;
  int nroots;

  set_timer_enabled(timing, 0 >= args.warmup);

  MPICHECK(MPI_Comm_size(mpicomm, &nranks));
  MPICHECK(MPI_Comm_rank(mpicomm, &rank));

  int initial_count = resize_count(nranks);
  color = pick_color(rank, nranks);

  if (initial_count == 0) return;
  if (initial_count == nranks) return;

  start_timer(timing, mpi_barrier);
  MPICHECK(MPI_Barrier(mpicomm));
  stop_timer(timing, mpi_barrier);

  pcclConfig_t config = PCCL_CONFIG_INITIALIZER;
  configure_communicator(&config);

  MPI_Comm mpi_color_comm;
  MPICHECK(MPI_Comm_split(mpicomm, color == 0 ? 0 : MPI_UNDEFINED, rank, &mpi_color_comm));
  timer_record_t* parent_timing = NULL;
  nroots = rootCount(initial_count);
  if (color == 0) {
    parent_timing = timing_allocate(mpi_color_comm, args.timing_flush_interval, "parent", nranks);
    set_timer_enabled(parent_timing, 0 >= args.warmup);
    all_timings.push_back(parent_timing);
    pcclUniqueId *id_array = (pcclUniqueId*)calloc(nroots, PCCL_UNIQUE_ID_BYTES);
    if (isRoot(rank)){
      PCCLCHECK(pcclGetUniqueId(&commId_smaller));
    }

    int* recv_count = (int*)malloc(sizeof(int) * initial_count);
    int* recv_displ = (int*)malloc(sizeof(int) * initial_count);
    int c=0;
    for (int i=0; i<initial_count; ++i){
      int peer_size = isRoot(i) ? PCCL_UNIQUE_ID_BYTES : 0;
      recv_displ[i] = c;
      recv_count[i] = peer_size;
      c+= peer_size;
    }
    start_timer(parent_timing, mpi_allgatherv);
    MPICHECK(MPI_Allgatherv(&commId_smaller,recv_count[rank],MPI_CHAR,id_array,recv_count,recv_displ,MPI_CHAR,mpi_color_comm));
    stop_timer(parent_timing, mpi_allgatherv);
    free(recv_count);
    free(recv_displ);

    start_timer(parent_timing, pccl_init);
    PCCLCHECK(pcclCommInitRankScalable(&pcclComm_smaller, initial_count, rank, nroots, id_array, &config));
    stop_timer(parent_timing, pccl_init);
    free(id_array);
  } else {
    pcclComm_smaller = NULL;
  }

  for (int i=0; i<args.work_iterations; i++) {
    do_work(parent_timing, pcclComm_smaller);
  }

  for (int j=0; j<args.iterations + args.warmup; j++) {
    set_timer_enabled(timing, j >= args.warmup);

    MPICHECK(MPI_Barrier(mpicomm));

    if (rank == 0) {
      // Rank 0 belongs to the parent communicator.
      PCCLCHECK(pcclCommGetUniqueId(pcclComm_smaller, &commId_bigger));
    }
    MPICHECK(MPI_Bcast(&commId_bigger, PCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, mpicomm));

    start_timer(timing, pccl_grow);
    if (color == 0) {
      PCCLCHECK(pcclCommGrow(pcclComm_smaller, nranks, &commId_bigger, -1, &pcclComm_bigger, &config));
    } else {
      PCCLCHECK(pcclCommGrow(NULL, nranks, &commId_bigger, rank, &pcclComm_bigger, &config));
    }
    stop_timer(timing, pccl_grow);

    for (int i=0; i<args.work_iterations; i++) {
      do_work(timing, pcclComm_bigger);
    }

    if (args.aborts) {
      start_timer(timing, pccl_abort);
      PCCLCHECK(pcclCommAbort(pcclComm_bigger));
      stop_timer(timing, pccl_abort);
    } else {
      start_timer(timing, pccl_finalize);
      PCCLCHECK(pcclCommFinalize(pcclComm_bigger));
      stop_timer(timing, pccl_finalize);
      start_timer(timing, pccl_destroy);
      PCCLCHECK(pcclCommDestroy(pcclComm_bigger));
      stop_timer(timing, pccl_destroy);
    }
  }

  MPICHECK(MPI_Barrier(mpicomm));
  if (color == 0) {
    set_timer_enabled(parent_timing, true);
    if (args.aborts) {
      start_timer(parent_timing, pccl_abort);
      PCCLCHECK(pcclCommAbort(pcclComm_smaller));
      stop_timer(parent_timing, pccl_abort);
    } else {
      start_timer(parent_timing, pccl_finalize);
      PCCLCHECK(pcclCommFinalize(pcclComm_smaller));
      stop_timer(parent_timing, pccl_finalize);
      start_timer(parent_timing, pccl_destroy);
      PCCLCHECK(pcclCommDestroy(pcclComm_smaller));
      stop_timer(parent_timing, pccl_destroy);
    }
  }
}

void timing_set_comm(timer_record_t* timing, MPI_Comm comm) {
  // Attach MPI after collecting pre-initialization timing events.
  if (timing->comm != MPI_COMM_NULL) {
    printf("ERROR: Timing already has a communicator!\n");
    ABORT_EXIT;
  }
  timing->comm = comm;
  MPICHECK(MPI_Comm_rank(comm, &timing->rank));
  MPICHECK(MPI_Comm_size(comm, &timing->rank_count));
  timing->sweep_size = timing->rank_count;
}

// All ranks must call stop_timer or skip_timer in the same order.
timer_record_t* timing_allocate(MPI_Comm comm, int flush, const char* role, int sweep_size, int color) {
  timer_record_t* timing = (timer_record_t*)malloc(sizeof(timer_record_t));
  timing->flush_interval = flush;
  timing->currently_enabled = true;
  timing->local = (double*)malloc(sizeof(double) * timing->flush_interval);
  timing->local_sq = (double*)malloc(sizeof(double) * timing->flush_interval);
  timing->id = (int*)malloc(sizeof(int) * timing->flush_interval);
  timing->participated = (int*)malloc(sizeof(int) * timing->flush_interval);
  timing->comm = MPI_COMM_NULL;
  timing->count = 0;
  timing->role = role;
  timing->color = color;

  if (comm != MPI_COMM_NULL) {
    timing_set_comm(timing, comm);
  } else {
    timing->rank = 0;
    timing->rank_count = 1;
  }
  timing->sweep_size = sweep_size ? sweep_size : timing->rank_count;
  if (timing->rank == 0) {
    timing->max = (double*)malloc(sizeof(double) * timing->flush_interval);
    timing->sum = (double*)malloc(sizeof(double) * timing->flush_interval);
    timing->sumsq = (double*)malloc(sizeof(double) * timing->flush_interval);
    timing->participant_count = (int*)malloc(sizeof(int) * timing->flush_interval);
  } else {
    timing->max = NULL;
    timing->sum = NULL;
    timing->sumsq = NULL;
    timing->participant_count = NULL;
  }

  for (int i=0; i<timer_enum_count; i++) {
    timing->rounds_by_type[i] = 0;
    timing->sum_by_type[i] = 0;
    timing->sumsq_by_type[i] = 0;
    timing->max_by_type[i] = 0;
    timing->participant_count_by_type[i] = 0;
  }
  return timing;
}

void timing_print_summary_grouped(timer_record_t** timings, int num_timings) {
  for (int k=0; k<num_timings; k++) {
    flush_timing(timings[k]);
  }
  for (int i=0; i<timer_enum_count; i++) {
    if (args.pccl_only && !timer_type_pccl(i)) continue;
    for (int k=0; k<num_timings; k++) {
      timer_record_t* timing = timings[k];
      if (timing->rank != 0) continue;
      int rounds_counted = timing->rounds_by_type[i];
      if (rounds_counted <= 0) continue;
      double avg = (double)timing->sum_by_type[i] / timing->participant_count_by_type[i];
      double stddev;
      if (timing->participant_count_by_type[i] > 1) {
        double A = timing->sumsq_by_type[i] / timing->participant_count_by_type[i];
        double B = pow(timing->sum_by_type[i] / timing->participant_count_by_type[i], 2.0);
        stddev = sqrt(std::max(0.0, A - B));
      } else {
        stddev = 0;
      }
      printf("SUMMARY: Comm size %4d, Role %7s, Color %d, Sweep size %4d, %36s: Max %8.3f ms, Avg %8.3f ms, Stddev %8.3f ms, Collectives %3d\n",
        timing->rank_count, timing->role, timing->color, timing->sweep_size,
        timer_strings[i], timing->max_by_type[i]*1e-6, avg*1e-6, stddev*1e-6,
        rounds_counted);
    }
  }
  if (args.root_every > 0 && num_timings && timings[0]->rank == 0) {
    printf("Note: Scalable Comm Initialization was performed with a root for every %d ranks.\n",args.root_every);
  }
}

void timing_print_summary(timer_record_t* timing) {
  timing_print_summary_grouped(&timing, 1);
}

void timing_free(timer_record_t** timing_ptr) {
  timer_record_t* timing = *timing_ptr;
  free(timing->local);
  free(timing->local_sq);
  free(timing->id);
  free(timing->participated);
  if (timing->max != NULL) free(timing->max);
  if (timing->sum != NULL) free(timing->sum);
  if (timing->sumsq != NULL) free(timing->sumsq);
  if (timing->participant_count != NULL) free(timing->participant_count);
  free(timing);
  *timing_ptr = NULL;
}

int next_step(int current) {
  double next;
  if (args.sweep_kind == kind_fixed) {
    return 0;
  }
  if (args.sweep_kind == kind_factor) {
    next = round(current * args.sweep_num);
  } else {
    next = current + args.sweep_num;
  }
  if (next == current) {
    next += (args.beginranks < args.endranks) ? 1 : -1;
  }
  if (next < 1 || next > std::max(args.beginranks, args.endranks)) return 0;
  return (int)next;
}

int keep_testing(int current) {
  if (current == 0) return 0;
  if (args.beginranks == args.endranks) {
    return current == args.endranks;
  }
  if (args.beginranks < args.endranks) {
    return current <= args.endranks;
  } else {
    return current >= args.endranks;
  }
}

int main(int argc, char **argv) {
  timer_record_t *timing_prempi;
  int rank, nDevs;

  std::vector<timer_record_t*> all_timings;

  // Buffer timing events until MPI is initialized.
  timing_prempi = timing_allocate(MPI_COMM_NULL, 16);

  start_timer(timing_prempi, mpi_init);
  MPICHECK(MPI_Init(&argc, &argv));
  stop_timer(timing_prempi, mpi_init);
  timing_set_comm(timing_prempi, MPI_COMM_WORLD);

  parse_args(argc, argv);

  start_timer(timing_prempi, hggc_init);
  HGGCCHECK(hggcGetDeviceCount(&nDevs));
  stop_timer(timing_prempi, hggc_init);

  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  MPI_Comm localcomm;
  int localrank;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localcomm));
  MPICHECK(MPI_Comm_rank(localcomm, &localrank));
  MPICHECK(MPI_Comm_free(&localcomm));
  const char* device_env = getenv("NCCL_TESTS_DEVICE");
  int device = device_env ? parse_integer(device_env, -1) : -1;
  if (device == -1) device = localrank;
  if (device >= nDevs) {
    fprintf(stderr, "Error: Selected PPU %d is unavailable (%d visible devices, local rank %d)\n", device, nDevs, localrank);
    ABORT_EXIT;
  }
  HGGCCHECK(hggcSetDevice(device));
  prepare_work(timing_prempi);

  all_timings.push_back(timing_prempi);

  if (rank == 0 && args.test_kind == test_shrink && args.share && args.aborts) {
    printf("# Shrink resource sharing is disabled with --abort.\n");
  }

  int tested_sizes = 0;
  int nrank_test = args.beginranks;
  while (keep_testing(nrank_test)) {
    int count = resize_count(nrank_test);
    if (args.test_kind != test_init && (count <= 0 || count >= nrank_test)) {
      if (rank == 0) printf("# Skipping comm size %d: resize does not produce two nonempty groups.\n", nrank_test);
      nrank_test = next_step(nrank_test);
      continue;
    }
    tested_sizes++;
    MPI_Comm subcomm = MPI_COMM_NULL;
    timer_record_t *subtimer;
    int color = 0;
    if (rank >= nrank_test) color = MPI_UNDEFINED;
    MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, color, rank, &subcomm));

    if (subcomm != MPI_COMM_NULL) {
      MPICHECK(MPI_Bcast(&work_ctx.seed, 1, MPI_UINT64_T, 0, subcomm));
      const char* role = args.test_kind == test_init ? "initial" : args.test_kind == test_grow ? "child" : "parent";
      subtimer = timing_allocate(subcomm, args.timing_flush_interval, role);
      all_timings.push_back(subtimer);
      if (args.test_kind == test_init) {
        main_init(argc, argv, subtimer, subcomm);
      } else if (args.test_kind == test_split) {
        main_split(argc, argv, subtimer, subcomm, all_timings);
      } else if (args.test_kind == test_shrink) {
        main_split(argc, argv, subtimer, subcomm, all_timings);
      } else if (args.test_kind == test_grow) {
        main_grow(argc, argv, subtimer, subcomm, all_timings);
      }
    }
    nrank_test = next_step(nrank_test);
  }

  cleanup_work();

  if (tested_sizes == 0) {
    if (rank == 0) printf("# No communicator sizes were tested.\n");
    main_ret_val = 1;
  }
  if (args.check) {
    int64_t wrong;
    uint64_t checked;
    MPICHECK(MPI_Allreduce(&work_ctx.total_wrong, &wrong, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD));
    MPICHECK(MPI_Allreduce(&work_ctx.total_checked, &checked, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
    if (checked) {
      if (rank == 0) printf("#wrong: %lld\n", (long long)wrong);
    } else {
      if (rank == 0) printf("# Verification skipped: no AllReduce results were checked.\n");
    }
    if (wrong || !checked) main_ret_val = 1;
  }

  timing_print_summary_grouped(all_timings.data(), all_timings.size());

  for (size_t i=0; i<all_timings.size(); i++) {
    if (all_timings[i]->comm != MPI_COMM_NULL && all_timings[i]->comm != MPI_COMM_WORLD) {
      MPICHECK(MPI_Comm_free(&all_timings[i]->comm));
    }
    timing_free(&all_timings[i]);
  }
  all_timings.clear();

  MPICHECK(MPI_Finalize());
  return main_ret_val;
}

void print_help() {
  int rank;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (rank == 0) {
    printf("Usage: comm_ops_perf <test> <test-options>\n");
    printf(" Tests: init, split, shrink, grow\n");
    printf("  init:   Time pcclCommInitRankScalable\n");
    printf("  split:  Time pcclCommSplit\n");
    printf("  shrink: Time pcclCommShrink\n");
    printf("  grow:   Time pcclCommGrow\n");
    printf(" Test Options:\n");
    printf("    -a|--abort: Don't finalize, instead abort communicators.\n");
    printf("          Shrink uses PCCL_SHRINK_ABORT, which disables resource sharing.\n");
    printf("    -A|--all-timers: Show timing for All operations, not just PCCL.\n");
    printf("    -c|--check <0/1>: Verify AllReduce results outside timed intervals. (Default: 1)\n");
    printf("    -s|--sweep-comms (factor-N|step-N|fixed) how to sweep through comm size. (Default: factor-2)\n");
    printf("    -b|--sweep-begin <ranks>: Ranks count to start with. (Default: NRanks)\n");
    printf("    -e|--sweep-end <ranks>: Ranks count to end with. (Default: 1)\n");
    printf("    -F|--flush-interval <interval>: flush timing every <interval> timing calls.\n");
    printf("          Default to 1, which introduces barriers and with -v flag, prints events as they happen.\n");
    printf("          Larger numbers remove artificial barriers, and prints in batches.\n");
    printf("    -i|--iterations <iterations>: number of iterations to run after warmup. (Default: %d).\n", args.iterations);
    printf("    -r|--resize factor-N|step-N:  How much to resize for split/shrink/grow.  (Default: factor-2)\n");
    printf("    -R|--root-every <n>: Each <n> ranks gets a root in pcclCommInitRankScalable (Default to 0 => only 1 root).\n");
    printf("    -S|--share: configure communicator with splitShare=1 and shrinkShare=1 (Default to %s).\n", args.share ? "true" : "false");
    printf("    -v|--verbose: print verbose timing output (timeseries).\n");
    printf("    -w|--warmup <iterations>: Run <iterations> rounds before enabling timing. (Default to %d).\n", args.warmup);
    printf("    -W|--work-iterations <iterations>: AllReduce calls per communicator. (Default: %d)\n", args.work_iterations);
    printf("Examples:\n");
    printf("  mpirun ./comm_ops_perf init\n");
    printf("  mpirun ./comm_ops_perf split --sweep-comms fixed\n");
    printf("  mpirun ./comm_ops_perf shrink\n");
    printf("  mpirun ./comm_ops_perf grow --resize factor-1.5\n");
    printf("For Detailed output: Add --all-timers and --verbose.\n");
    printf("Note begin/end ranks can be increasing or decreasing, and the comm sweep will be adjusted accordingly.\n");
    printf("Sizes without two nonempty resize groups are skipped; a run with no tested sizes fails.\n");
    printf("AllReduce timings measure the host call; stream synchronization is reported separately with --all-timers.\n");
    printf("NCCL_TESTS_DEVICE selects a visible device index; -1 selects by local MPI rank.\n");
    printf("\n");
  }
}

#else
int main() {
  printf("This test requires PCCL 2.3.0 or later.\n");
  return 1;
}
#endif
#endif
