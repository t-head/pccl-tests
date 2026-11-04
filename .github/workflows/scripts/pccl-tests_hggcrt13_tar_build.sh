#!/bin/bash
set -e
# Validate required environment variables.
if [ -z "${source_home}" ]; then
    echo "Error: source_home environment variable is not set"
    exit 1
fi
if [ -z "${build_output}" ]; then
    echo "Error: build_output environment variable is not set"
    exit 1
fi

# Matrix builds override pkg_name/sdk_package so every distribution keeps a distinct
# tarball name and installs the SDK build that matches its own libc/toolchain.
pkg_name=${pkg_name:-pccl_hggcrt13_afa3}
echo "######################### $(date +%Y%m%d%H%M%S) Source PPU_SDK ##########################"

source /usr/local/PPU_SDK/envsetup.sh
cd ${source_home}/pccl
if [ -n "${package_version}" ]; then
    echo "Building with package version: ${package_version}"
    bash build.sh --use_sdk --hggcrt_version 13 --rebuild --package_version ${package_version}
else
    echo "Building without package version (default)"
    VERSION_FILE="${source_home}/pccl/makefiles/version.mk"
    PCCL_MAJOR=$(grep "^PCCL_MAJOR" ${VERSION_FILE} | sed 's/.*:=\s*//')
    PCCL_MINOR=$(grep "^PCCL_MINOR" ${VERSION_FILE} | sed 's/.*:=\s*//')
    PCCL_PATCH=$(grep "^PCCL_PATCH" ${VERSION_FILE} | sed 's/.*:=\s*//')
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    GENERATED_VERSION="${PCCL_MAJOR}.${PCCL_MINOR}.${PCCL_PATCH}-${TIMESTAMP}"
    bash build.sh --use_sdk --hggcrt_version 13 --rebuild --package_version ${GENERATED_VERSION}
fi

export PCCL_HOME=${source_home}/pccl/build

mkdir -p ${build_output}/comm_internal/comm_tools/single_process
mkdir -p ${build_output}/comm_release/comm_tools/single_process
mkdir -p ${build_output}/comm_internal/comm_tools/multi_process
mkdir -p ${build_output}/comm_release/comm_tools/multi_process
mkdir -p ${build_output}/comm_internal/comm_tools/licenses
mkdir -p ${build_output}/comm_release/comm_tools/licenses

echo "BUILD single_process"
cd ${source_home}/pccl-tests
bash build.sh --rebuild

cp ${source_home}/pccl-tests/build/all_gather_perf ${build_output}/comm_internal/comm_tools/single_process
cp ${source_home}/pccl-tests/build/all_reduce_perf ${build_output}/comm_internal/comm_tools/single_process
cp ${source_home}/pccl-tests/build/broadcast_perf ${build_output}/comm_internal/comm_tools/single_process
cp ${source_home}/pccl-tests/build/reduce_scatter_perf ${build_output}/comm_internal/comm_tools/single_process
cp ${source_home}/pccl-tests/build/reduce_perf ${build_output}/comm_internal/comm_tools/single_process
cp ${source_home}/pccl-tests/build/alltoall_perf ${build_output}/comm_internal/comm_tools/single_process
cp ${source_home}/pccl-tests/build/gather_perf ${build_output}/comm_internal/comm_tools/single_process
cp ${source_home}/pccl-tests/build/scatter_perf ${build_output}/comm_internal/comm_tools/single_process
cp ${source_home}/pccl-tests/build/sendrecv_perf ${build_output}/comm_internal/comm_tools/single_process
cp ${source_home}/pccl-tests/build/hypercube_perf ${build_output}/comm_internal/comm_tools/single_process

cp ${source_home}/pccl-tests/build/all_gather_perf ${build_output}/comm_release/comm_tools/single_process
cp ${source_home}/pccl-tests/build/all_reduce_perf ${build_output}/comm_release/comm_tools/single_process
cp ${source_home}/pccl-tests/build/broadcast_perf ${build_output}/comm_release/comm_tools/single_process
cp ${source_home}/pccl-tests/build/reduce_scatter_perf ${build_output}/comm_release/comm_tools/single_process
cp ${source_home}/pccl-tests/build/reduce_perf ${build_output}/comm_release/comm_tools/single_process
cp ${source_home}/pccl-tests/build/alltoall_perf ${build_output}/comm_release/comm_tools/single_process
cp ${source_home}/pccl-tests/build/gather_perf ${build_output}/comm_release/comm_tools/single_process
cp ${source_home}/pccl-tests/build/scatter_perf ${build_output}/comm_release/comm_tools/single_process
cp ${source_home}/pccl-tests/build/sendrecv_perf ${build_output}/comm_release/comm_tools/single_process
cp ${source_home}/pccl-tests/build/hypercube_perf ${build_output}/comm_release/comm_tools/single_process

echo "BUILD multi_process"
cd ${source_home}/pccl-tests
bash build.sh --rebuild --use_mpi

cp ${source_home}/pccl-tests/build/all_gather_perf ${build_output}/comm_internal/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/all_reduce_perf ${build_output}/comm_internal/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/broadcast_perf ${build_output}/comm_internal/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/reduce_scatter_perf ${build_output}/comm_internal/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/reduce_perf ${build_output}/comm_internal/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/alltoall_perf ${build_output}/comm_internal/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/gather_perf ${build_output}/comm_internal/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/scatter_perf ${build_output}/comm_internal/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/sendrecv_perf ${build_output}/comm_internal/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/hypercube_perf ${build_output}/comm_internal/comm_tools/multi_process

cp ${source_home}/pccl-tests/build/all_gather_perf ${build_output}/comm_release/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/all_reduce_perf ${build_output}/comm_release/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/broadcast_perf ${build_output}/comm_release/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/reduce_scatter_perf ${build_output}/comm_release/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/reduce_perf ${build_output}/comm_release/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/alltoall_perf ${build_output}/comm_release/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/gather_perf ${build_output}/comm_release/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/scatter_perf ${build_output}/comm_release/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/sendrecv_perf ${build_output}/comm_release/comm_tools/multi_process
cp ${source_home}/pccl-tests/build/hypercube_perf ${build_output}/comm_release/comm_tools/multi_process

cp ${source_home}/pccl-tests/LICENSE.txt ${build_output}/comm_internal/comm_tools/licenses/nccl-tests-LICENSE.txt
cp ${source_home}/pccl-tests/LICENSE.txt ${build_output}/comm_release/comm_tools/licenses/nccl-tests-LICENSE.txt

cd ${build_output}/comm_internal/ 
tar -zcvf ${build_output}/comm_internal/"${pkg_name}.tar.gz" *
ls | grep -v "${pkg_name}.tar.gz" | xargs rm -rf
cd ${build_output}/comm_release/ 
tar -zcvf ${build_output}/comm_release/"${pkg_name}.tar.gz" *
ls | grep -v "${pkg_name}.tar.gz" | xargs rm -rf
echo "BUILD PCCL FINISHED"