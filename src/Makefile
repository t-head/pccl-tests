#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for license information
#

# PPU_ARCHS can be any combination of ppu_10,ppu_15
PPU_ARCHS ?= ppu_10,ppu_15
VERBOSE ?= 0
DEBUG ?= 0
PPU_SDK ?= /usr/local/PPU_SDK
HGCC ?= $(PPU_SDK)/bin/hgcc
PPU_LIB ?= $(PPU_SDK)/lib
PPU_INC ?= $(PPU_SDK)/include
HGGC_VERSION = $(strip $(shell which $(HGCC) >/dev/null && $(HGCC) --version | grep "version" | sed 's/.*version //' | sed 's/ .*//'))
HGGC_MAJOR = $(shell echo $(HGGC_VERSION) | cut -d "." -f 1)
HGGC_MINOR = $(shell echo $(HGGC_VERSION) | cut -d "." -f 2)

HGFLAGS    := -std=c++17 -x hg -Xcompiler -fPIC -DOMPI_SKIP_MPICXX

ifneq ($(PCCL_HOME),)
  HGFLAGS += -I$(PCCL_HOME)/include/
  HGLDFLAGS += -L$(PCCL_HOME)/lib -Xlinker -L=$(PCCL_HOME)/lib \
               -Xlinker -rpath-link=$(PCCL_HOME)/lib \
               -Xlinker -rpath=$(PCCL_HOME)/lib
  PCCL_LIB := $(PCCL_HOME)/lib/libpccl.so
endif

HGGCRT_LDFLAG = -lhggc_wrapper -lhg_wrapper
HGFLAGS += -isystem $(PPU_INC)
HGLDFLAGS += $(PPU_LIB:%=-L%)
HGLDFLAGS += $(PPU_LIB:%=-Xlinker -rpath-link=%)
ifneq ($(PCCL_LIB),)
  HGLDFLAGS += $(HGGCRT_LDFLAG) $(PCCL_LIB) -lpthread
else
  HGLDFLAGS += $(HGGCRT_LDFLAG) -lpccl -lpthread
endif

define append_arch
$(eval HGFLAGS += -arch=$(1))
endef

SPLIT_ARCHS = $(shell echo $(PPU_ARCHS) | tr ',' ' ')
TMP := $(foreach arch,$(SPLIT_ARCHS), $(call append_arch,$(arch)))

ifeq ($(DEBUG), 0)
  HGFLAGS += -O3 -dopt 3
else
  HGFLAGS += -O0 -dopt 0 -G -g -lineinfo
endif

ifneq ($(VERBOSE), 0)
  HGFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
else
.SILENT:
endif

ifeq ($(MPI), 1)
  MPI_INCDIRS := $(shell mpicc --showme:incdirs)
  MPI_LIBDIRS := $(shell mpicc --showme:libdirs)
  HGFLAGS += -DMPI_SUPPORT $(addprefix -I,$(MPI_INCDIRS))
  HGLDFLAGS += $(addprefix -L,$(MPI_LIBDIRS)) -lmpi
endif

ifneq ($(USE_SIMT), 1)
  HGFLAGS += --no-simt
endif

ifeq ($(KEEP), 1)
  HGFLAGS += -keep -lineinfo
endif

HGFLAGS += $(shell echo "$(PPU_OPTION)" | sed "s/;/ /g")

.PHONY: build clean
.DEFAULT_GOAL := build

BUILDDIR ?= ../build
DST_DIR := $(BUILDDIR)

FORCE:
.PHONY: FORCE

COMMON_SRC_FILES := common.cu barrier.cu

ifeq ($(DISABLE_HGPTI),1)
  HGFLAGS += -DDISABLE_HGPTI
else
  HGLDFLAGS += -lhgpti
  COMMON_SRC_FILES += hgpti_common.cu
endif

COMMON_OBJ_FILES := $(COMMON_SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce alltoall alltoallv sendrecv scatter gather hypercube

PCCL_HEADER := $(firstword $(wildcard $(PCCL_HOME)/include/pccl.h $(PPU_INC)/pccl.h /usr/local/include/pccl.h /usr/include/pccl.h))
COMM_OPS_MIN_PCCL_VERSION := 2300
COMM_OPS_PCCL_VERSION := $(shell test -n "$(PCCL_HEADER)" && \
  awk '/^\#define PCCL_(MAJOR|MINOR|PATCH) /{v[$$2]=$$3} END{if (v["PCCL_MAJOR"]!="") print v["PCCL_MAJOR"]*1000+v["PCCL_MINOR"]*100+v["PCCL_PATCH"]}' "$(PCCL_HEADER)")
COMM_OPS_SUPPORTED := $(shell test -n "$(COMM_OPS_PCCL_VERSION)" && \
  test "$(COMM_OPS_PCCL_VERSION)" -ge $(COMM_OPS_MIN_PCCL_VERSION) && echo 1)
ifeq ($(MPI), 1)
ifeq ($(COMM_OPS_SUPPORTED), 1)
  BIN_FILES_LIST += comm_ops
else
  $(warning Skipping comm_ops_perf: PCCL 2.3.0 or later headers are required)
endif
endif

BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

info:
	@printf '%s\n' \
		'=================================================' \
		'  PPU_ARCHS      : $(PPU_ARCHS)'                   \
		'  HGGC_VERSION   : $(HGGC_VERSION)'                \
		'  HGCC           : $(HGCC)'                        \
		'=================================================' \

build: info ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}
	rm -f *.o *.s *.hgfb* *.hggci *.lk *.bc a.out* *hggc* *splitter* *host*

# GIN device API benchmarks; they add themselves to "build" when MPI=1.
include device_api/gin/gin.mk

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILDDIR)/verifiable
include ../verifiable/verifiable.mk

${DST_DIR}/%.o: %.cu common.h $(TEST_VERIFIABLE_HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(HGCC) -o $@ $(HGFLAGS) -c $<

${DST_DIR}/%_perf:${DST_DIR}/%.o ${COMMON_OBJ_FILES}  $(TEST_VERIFIABLE_OBJS)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(HGCC) -o $@ $^ ${HGLDFLAGS}

ifneq ($(MPI), 1)
${DST_DIR}/comm_ops_perf:
	$(error Communicator operations test requires MPI=1)
else ifneq ($(COMM_OPS_SUPPORTED), 1)
${DST_DIR}/comm_ops_perf:
	$(error Communicator operations test requires PCCL 2.3.0 or later headers)
else
${DST_DIR}/comm_ops_perf: ${DST_DIR}/comm_ops.o $(TEST_VERIFIABLE_OBJS)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(HGCC) -o $@ $^ ${HGLDFLAGS}
endif
