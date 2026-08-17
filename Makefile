#
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENCE.txt for license information
#

.PHONY : all clean

default : src.build

TARGETS=src

all:   ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build

%.clean:
	${MAKE} -C $* clean
