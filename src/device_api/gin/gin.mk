#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for license information
#

# Each benchmark category (latency and throughput) has runner sources
# (*_main.cu) and operation sources; the operation objects are archived once
# per category and linked into every category runner.  Binaries are written to
# $(BUILDDIR)/device_api/gin/<category>/<benchmark>_perf.
DEVICE_API_GIN_SRCDIR       := $(abspath device_api/gin)
DEVICE_API_GIN_UTILS_SRCDIR := $(DEVICE_API_GIN_SRCDIR)/utils
DEVICE_API_GIN_UTILS_DST    := $(DST_DIR)/device_api/gin/utils
DEVICE_API_GIN_MPI_ENABLED  := $(filter 1,$(MPI))
# The utils directory comes first so that "common.h" resolves to the GIN
# harness header and not to the collective framework header in src/.
DEVICE_API_GIN_EXTRA_INC := -I$(DEVICE_API_GIN_UTILS_SRCDIR) -I$(abspath .)

# GIN throughput tests support 1024-thread CTAs, which require at most 60
# registers per thread. Change to use these guidelines -> threads:regs
# 512 : 120, 640 : 96, 768 : 80, 1024 : 60
DEVICE_API_GIN_HGFLAGS := $(filter-out -maxrregcount=%,$(HGFLAGS)) -maxrregcount=60

DEVICE_API_GIN_HGFLAGS := $(filter-out --no-simt,$(DEVICE_API_GIN_HGFLAGS)) -DPCCL_TESTS_GIT_VERSION=\"$(PCCL_TESTS_GIT_VERSION)\"

DEVICE_API_GIN_MIN_PCCL_VERSION := 2200
DEVICE_API_GIN_PCCL_H := $(firstword $(wildcard $(if $(PCCL_HOME),$(PCCL_HOME)/include/pccl.h) $(PPU_INC)/pccl.h))
DEVICE_API_GIN_PCCL_VERSION := $(shell test -n "$(DEVICE_API_GIN_PCCL_H)" && \
  awk '/^\#define PCCL_(MAJOR|MINOR|PATCH) /{v[$$2]=$$3} END{if (v["PCCL_MAJOR"]!="") print v["PCCL_MAJOR"]*1000+v["PCCL_MINOR"]*100+v["PCCL_PATCH"]}' $(DEVICE_API_GIN_PCCL_H))
DEVICE_API_GIN_PCCL_TOO_OLD := $(shell test -n "$(DEVICE_API_GIN_PCCL_VERSION)" && \
  test $(DEVICE_API_GIN_PCCL_VERSION) -ge $(DEVICE_API_GIN_MIN_PCCL_VERSION) && echo 0 || echo 1)

ifneq ($(DEVICE_API_GIN_MPI_ENABLED),)
ifeq ($(DEVICE_API_GIN_PCCL_TOO_OLD),1)
$(warning Skipping GIN device API performance tests: pccl version code '$(DEVICE_API_GIN_PCCL_VERSION)' ($(DEVICE_API_GIN_PCCL_H)) is older than $(DEVICE_API_GIN_MIN_PCCL_VERSION))
endif
endif

DEVICE_API_GIN_UTILS_SRCS := $(wildcard $(DEVICE_API_GIN_UTILS_SRCDIR)/*.cu)
DEVICE_API_GIN_UTILS_OBJS := $(patsubst $(DEVICE_API_GIN_UTILS_SRCDIR)/%.cu,$(DEVICE_API_GIN_UTILS_DST)/%.o,$(DEVICE_API_GIN_UTILS_SRCS))
DEVICE_API_GIN_UTILS_HDRS := $(wildcard $(DEVICE_API_GIN_UTILS_SRCDIR)/*.h) $(abspath pccl_tests_version.h)

DEVICE_API_GIN_CONFIG := $(DEVICE_API_GIN_UTILS_DST)/build_config.stamp
$(DEVICE_API_GIN_CONFIG): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' '$(HGCC) $(DEVICE_API_GIN_EXTRA_INC) $(DEVICE_API_GIN_HGFLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

$(DEVICE_API_GIN_UTILS_OBJS): $(DEVICE_API_GIN_UTILS_DST)/%.o: $(DEVICE_API_GIN_UTILS_SRCDIR)/%.cu $(DEVICE_API_GIN_UTILS_HDRS) $(DEVICE_API_GIN_CONFIG)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(DEVICE_API_GIN_UTILS_DST)
	$(HGCC) -o $@ $(DEVICE_API_GIN_EXTRA_INC) $(DEVICE_API_GIN_HGFLAGS) -c $<

# common.o prints the pccl-tests git revision.
$(DEVICE_API_GIN_UTILS_DST)/common.o: ${DST_DIR}/include/git_version.stamp

# Allow a single benchmark to be requested with the repository-relative path,
# e.g. make -C src build/device_api/gin/throughput/ginPutBW_perf MPI=1.
build/device_api/gin/%: $(DST_DIR)/device_api/gin/%
	@:

define DEVICE_API_GIN_CATEGORY
DEVICE_API_GIN_$(1)_SRCDIR := $$(DEVICE_API_GIN_SRCDIR)/$(1)
DEVICE_API_GIN_$(1)_DST := $$(DST_DIR)/device_api/gin/$(1)
DEVICE_API_GIN_$(1)_MAIN_SRCS := $$(wildcard $$(DEVICE_API_GIN_$(1)_SRCDIR)/*_main.cu)
DEVICE_API_GIN_$(1)_TEST_SRCS := $$(filter-out $$(DEVICE_API_GIN_$(1)_MAIN_SRCS),$$(wildcard $$(DEVICE_API_GIN_$(1)_SRCDIR)/*.cu))
DEVICE_API_GIN_$(1)_TEST_OBJS := $$(patsubst $$(DEVICE_API_GIN_$(1)_SRCDIR)/%.cu,$$(DEVICE_API_GIN_$(1)_DST)/%.o,$$(DEVICE_API_GIN_$(1)_TEST_SRCS))
DEVICE_API_GIN_$(1)_MAIN_OBJS := $$(patsubst $$(DEVICE_API_GIN_$(1)_SRCDIR)/%.cu,$$(DEVICE_API_GIN_$(1)_DST)/%.o,$$(DEVICE_API_GIN_$(1)_MAIN_SRCS))
DEVICE_API_GIN_$(1)_LIB := $$(DEVICE_API_GIN_$(1)_DST)/libginPerf$(1).a
DEVICE_API_GIN_$(1)_BINS := $$(patsubst $$(DEVICE_API_GIN_$(1)_SRCDIR)/%_main.cu,$$(DEVICE_API_GIN_$(1)_DST)/%_perf,$$(DEVICE_API_GIN_$(1)_MAIN_SRCS))
DEVICE_API_GIN_$(1)_HDRS := $$(wildcard $$(DEVICE_API_GIN_$(1)_SRCDIR)/*.h) $$(DEVICE_API_GIN_UTILS_HDRS)

ifneq ($$(DEVICE_API_GIN_MPI_ENABLED),)
ifneq ($$(DEVICE_API_GIN_PCCL_TOO_OLD),1)
build: $$(DEVICE_API_GIN_$(1)_BINS)
endif
endif

# Listing the targets up front makes these static pattern rules, which count as
# explicit and so always win over the top-level $${DST_DIR}/%.o rule.
$$(DEVICE_API_GIN_$(1)_TEST_OBJS) $$(DEVICE_API_GIN_$(1)_MAIN_OBJS): $$(DEVICE_API_GIN_$(1)_DST)/%.o: $$(DEVICE_API_GIN_$(1)_SRCDIR)/%.cu $$(DEVICE_API_GIN_$(1)_HDRS) $$(DEVICE_API_GIN_CONFIG)
	@printf "Compiling  %-35s > %s\n" $$< $$@
	@mkdir -p $$(DEVICE_API_GIN_$(1)_DST)
	$$(HGCC) -o $$@ $$(DEVICE_API_GIN_EXTRA_INC) $$(DEVICE_API_GIN_HGFLAGS) -c $$<

$$(DEVICE_API_GIN_$(1)_LIB): $$(DEVICE_API_GIN_$(1)_TEST_OBJS)
	@printf "Archiving  %-35s > %s\n" $(1) $$@
	@mkdir -p $$(DEVICE_API_GIN_$(1)_DST)
	rm -f $$@
	$$(AR) rcs $$@ $$^

ifeq ($$(DEVICE_API_GIN_MPI_ENABLED),)
$$(DEVICE_API_GIN_$(1)_BINS):
	$$(error GIN device API performance tests require MPI=1)
else ifeq ($$(DEVICE_API_GIN_PCCL_TOO_OLD),1)
$$(DEVICE_API_GIN_$(1)_BINS):
	$$(error GIN device API performance tests require pccl version code $$(DEVICE_API_GIN_MIN_PCCL_VERSION) or newer)
else
# The archive must follow the objects that reference it so the linker resolves
# the launch entry points the runner names, and takes only those members.
$$(DEVICE_API_GIN_$(1)_BINS): $$(DEVICE_API_GIN_$(1)_DST)/%_perf: $$(DEVICE_API_GIN_$(1)_DST)/%_main.o $$(DEVICE_API_GIN_$(1)_LIB) $$(DEVICE_API_GIN_UTILS_OBJS)
	@printf "Linking  %-35s > %s\n" $$< $$@
	@mkdir -p $$(DEVICE_API_GIN_$(1)_DST)
	$$(HGCC) -o $$@ $$< $$(DEVICE_API_GIN_UTILS_OBJS) $$(DEVICE_API_GIN_$(1)_LIB) $$(HGLDFLAGS)
endif
endef

$(eval $(call DEVICE_API_GIN_CATEGORY,latency))
$(eval $(call DEVICE_API_GIN_CATEGORY,throughput))
