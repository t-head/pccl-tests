# SPDX-FileCopyrightText: Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for license information

#!/bin/bash

set -e

ppu_archs="ppu_10,ppu_15"
hggcrt_version=12
use_mpi=0
rebuild=0
dry_run=0
debug=0
keep=0
verbose=0
ppu_option=""
disable_hgpti=0
cpu_num=0

function usage
{
    cat <<EOF
Usage: ./$(basename "$0") [OPTIONS]

Options:
    --ppu_archs <string>      Specify the PPU Archs, any combination of (ppu_10|ppu_15). (default: ppu_10,ppu_15)
    --use_mpi                 Specify to build multi-process pccl-tests. (boolean flag)
    --rebuild                 Specify to make clean and remove build directory before building pccl-tests. (boolean flag)
    --dry_run                 Specify to show what commands would be executed without actually running them. (boolean flag)
    --debug                   Specify to build pccl-tests with debug flag. (boolean flag)
    --keep                    Specify to retain all intermediate files generated during compilation.
    --verbose                 Specify to show intermediate build log. (boolean flag)
    --ppu_option              Specify compiler optimize and bug fix options. (default: "")
    --disable_hgpti <0,1>     Specify to build without hgpti. (boolean flag)
    --cpu_num <int>           Specify the number of parallel jobs for make -j (default: use all CPUs).
EOF
    exit 0
}

function build_test
{
    if [ $rebuild -eq 1 ]; then
        make clean
    fi
    if [ $cpu_num -gt 0 ]; then
        build_cmd_str="make -j${cpu_num}"
    else
        build_cmd_str="make -j"
    fi
    if [ $dry_run -eq 1 ]; then
        build_cmd_str="${build_cmd_str} --dry-run"
    fi
    build_cmd_str="${build_cmd_str} src.build PPU_ARCHS=${ppu_archs}"
    if [ $use_mpi -eq 1 ]; then
        build_cmd_str="${build_cmd_str} MPI=1"
    fi
    if [ $keep -eq 1 ]; then
        build_cmd_str="${build_cmd_str} KEEP=1"
    fi
    if [ $debug -eq 1 ]; then
        build_cmd_str="${build_cmd_str} DEBUG=1"
    fi
    if [ $verbose -eq 1 ]; then
        build_cmd_str="${build_cmd_str} VERBOSE=1"
    fi
    if [ $disable_hgpti -eq 1 ]; then
        build_cmd_str="${build_cmd_str} DISABLE_HGPTI=1"
    fi
    if [ "$ppu_option" != "" ]; then
        build_cmd_str="${build_cmd_str} PPU_OPTION=${ppu_option}"
    fi

    # build command executing
    $build_cmd_str
}

while [[ $# -ge 1 ]]
do
    key="$1"
    case $key in
        --debug)
            debug=1
            ;;
        --keep)
            keep=1
            ;;
        --use_mpi)
            use_mpi=1
            ;;
        --rebuild)
            rebuild=1
            ;;
        --dry_run)
            dry_run=1
            ;;
        --ppu_option)
            ppu_option="$2"
            shift
            ;;
        --ppu_archs)
            ppu_archs="$2"
            shift
            ;;
        --hggcrt_version)
            hggcrt_version="$2"
            shift
            ;;
        --disable_hgpti)
            disable_hgpti="$2"
            shift
            ;;
        --verbose)
            verbose=1
            ;;
        --cpu_num)
            cpu_num="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
    shift
done

build_test
