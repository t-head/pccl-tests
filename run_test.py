import os
import argparse
import string
import subprocess
import signal
import datetime
import sys
import re
import itertools
import copy
import shlex
from run_test_config import *

# TestCase fields, used to set TestCase class attr
FIELDS = {'op': None,
          'algo': None,
          'proto': None,
          'ranks': None,
          'data_type': None,
          'data_size': None,
          'iters': None,
          'threads': None,
          'nprocs': None,
          'nnodes': None,
          'redop': 'sum',
          'min_data_size': None,
          'max_data_size': None,
          'random_size_num': None,
          'average': None,
          'register': None,
          'cta_policy': None,
          'warmup': None,
          'init_type': None,
          }


class TestCase:
    """This class is used to construct a test case.

    If you want to generate test cases from list, please use 'gen_product_zip_cases' function.
    """

    def __init__(self, args, **kwargs):
        """construct test case from kwargs and args."""

        field_values = FIELDS

        merged = {}
        for name, value in field_values.items():
            if isinstance(value, (list, dict)):
                merged[name] = type(value)(value)
            else:
                merged[name] = value

        merged.update(kwargs)
        for name, value in merged.items():
            setattr(self, name, value)
        TestCase.fill_none_fields_from_args(args, self)

    @staticmethod
    def gen_product_zip_cases(cases_dict, case_list, args, zip_groups=[[]]):
        """This function generates test cases using the Cartesian product and put them in case_list.

        The zip_groups parameter specifies which fields to zip together.
        Multiple fields can be combined, and there can be multiple different combinations.
        An example:
        cases_dict = {  'op': ['all_gather_perf','all_reduce_perf'],
                        'algo' : ['Ring','Tree'],
                        'proto': ['Auto'],
                        'redop': ['sum'],
                        'ranks' : [2, 4, 8, 16, 32],
                        'data_type' : ['int8'],
                        'data_size' : ['32B', '32KB', '8MB', '32MB', '256MB']
                    }
        zip_groups = [['op', 'algo'], ['ranks', 'data_size']]
        """
        for k, v in cases_dict.items():
            if not isinstance(cases_dict[k], list):
                cases_dict[k] = [v]
        TestCase.caselist_filter_rule(cases_dict, args)
        zip_keys = set()
        product_data = {}
        zip_list = []
        for group in zip_groups:
            keys = group
            for k in keys:
                zip_keys.add(k)
            values = [cases_dict[k] for k in keys]
            group_combo = []
            for combo in zip(*values):
                group_combo.append({key: val for key, val in zip(keys, combo)})
            zip_list.append(group_combo)

        for key in cases_dict:
            if key not in zip_keys:
                product_data[key] = cases_dict[key]
        product_combos = []
        product_keys = list(product_data.keys())
        product_values = list(product_data.values())
        if len(zip_groups[0]) < 1:
            for combo in itertools.product(*product_values):
                product_case = TestCase(args,
                                        **{key: val for key, val in zip(product_keys, combo)})
                if TestCase.is_valid(product_case, args):
                    case_list.append(product_case)
            return
        for combo in itertools.product(*product_values):
            product_case = {key: val for key, val in zip(product_keys, combo)}
            product_combos.append(product_case)

        for merged_zip in itertools.product(*zip_list):
            merged_case = {}
            for case in merged_zip:
                merged_case.update(copy.deepcopy(case))
            for prod_case in product_combos:
                full_case = copy.deepcopy(merged_case)
                full_case.update(copy.deepcopy(prod_case))
                real_case = TestCase(args, **full_case)
                if TestCase.is_valid(real_case, args):
                    case_list.append(real_case)

    @staticmethod
    def is_valid(case, args):
        """verify case."""
        if case.nnodes * case.nprocs * case.threads > case.ranks:
            if args.case_dump < 1:
                print("'ranks' is not enough.")
            return False
        if case.ranks % case.nnodes != 0:
            print("case.ranks %% case.nnodes != 0")
            return False
        if case.ranks // case.nnodes > 16:
            print("ranks per node > 16.")
            return False
        if case.nnodes > 1:
            # filter-out unsupported protocols for cross node case.
            if 'Signal' in case.proto:
                print("unsupported protocols for cross node case: Signal.")
                return False
            # LL128 may be used on PHB path for Tree
            if case.nnodes > 2 and case.proto == 'LL128' and case.algo == 'Tree':
                print("unsupported cross node case, LL128 may be used on PHB path for Tree")
                return False
            # unsupport alltoall + Bulk for RDMA cross node
            if "Bulk" in case.proto and case.op == 'alltoall_perf':
                print("unsupport cross node case: alltoall + Bulk for RDMA")
                return False
        return True

    @staticmethod
    def caselist_filter_rule(caselist_dict, args):
        """filter in op value in args.op and filter out value in ignore_xxx_list."""
        arg_dict = vars(args)
        ignore_names = [name for name in arg_dict if name.startswith(
            'ignore_') and name.endswith('_list')]
        # filter-in params
        op_list = caselist_dict['op']
        op_list = [op for op in op_list if op in args.op]
        caselist_dict['op'] = op_list
        # filter out
        for name in ignore_names:
            ignore_list = arg_dict[name]
            param_name = name[7:-5]
            if caselist_dict.get(param_name) is not None:
                origin_list = caselist_dict[param_name]
                caselist_dict[param_name] = [
                    val for val in origin_list if val not in ignore_list]

    @staticmethod
    def get_case_dict_from_args(args):
        """generate case_dict from args."""
        case_dict = {}
        for name in FIELDS:
            if getattr(args, name, None) is not None:
                case_dict[name] = getattr(args, name)
        return case_dict

    @staticmethod
    def fill_none_fields_from_args(args, case):
        """Fill none fields from args. If args are not set, use the default value"""
        for name, _ in FIELDS.items():
            val = getattr(case, name, None)
            arg_val = getattr(args, name, None)
            if val is None:
                if not isinstance(arg_val, list):
                    setattr(case, name, arg_val)
                else:
                    setattr(case, name, arg_val[0])

    @staticmethod
    def is_subset(case, case_dict):
        """Returns True if the case's associated field value is in the case_dict, otherwise returns False"""
        for key in case_dict.keys():
            if getattr(case, key) not in case_dict[key]:
                return False
        return True


CaseList = []


def apply_env_args_rule(args):
    """set env according to args."""
    # Export visible devices
    if args.devices != '':
        os.environ['HGGC_VISIBLE_DEVICES'] = args.devices
    # Enable pccl kernels launching mode as group launch
    if args.group_launch:
        os.environ['PCCL_LAUNCH_MODE'] = 'GROUP'

    if args.launch_mode == 'k8s':
        if args.cross_node:
            if args.cross_node == 'socket':
                os.environ['PCCL_IB_DISABLE'] = '1'
            elif args.cross_node == 'icn_scale_out':
                os.environ['PCCL_ICN_SCALE_OUT_DISABLE'] = '0'
            else:
                os.environ['PCCL_IB_DISABLE'] = '0'
                os.environ['PCCL_IB_QPS_PER_CONNECTION'] = '8'
                if args.cross_node == 'gdr':
                    os.environ['PCCL_NET_GDR_LEVEL'] = 'SYS'
                else:
                    os.environ['PCCL_NET_GDR_LEVEL'] = 'LOC'
        os.environ['PCCL_BLOCKING_BOOTSTRAP'] = '1'
    os.environ['PCCL_SOCKET_IFNAME'] = '^flannel,docker,lo'
    if args.use_4nics:
        os.environ['PCCL_IB_HCA'] = 'mlx5_bond_1,mlx5_bond_3,mlx5_bond_5,mlx5_bond_7'
    if args.hw_perf:
        os.environ['PCCL_DEBUG_FILE'] = 'pccl_perflog-{}.log'.format(
            os.getenv('RANK'))
    if args.profile == 'acu':
        os.environ['PCCL_LAUNCH_MODE'] = 'PARALLEL'
    os.environ['PATH'] = "{}:{}".format(os.environ['PATH'], os.path.realpath("./build"))


def apply_env_case_rule(args, case):
    """set env according to args and case info."""
    # pop case evn
    os.environ.pop('PCCL_ALGO', None)
    os.environ.pop('PCCL_DISABLE_FC_GRAPH', None)
    os.environ.pop('PCCL_PROTO', None)
    os.environ.pop('PCCL_LL_BUFFSIZE', None)
    os.environ.pop('PCCL_LL128_BUFFSIZE', None)
    os.environ.pop('PCCL_BUFFSIZE', None)
    os.environ.pop('PCCL_SIGNAL_BUFFSIZE', None)
    os.environ.pop('PCCL_SIMPLE_BULK_BUFFSIZE', None)
    os.environ.pop('PCCL_SIGNAL_BULK_BUFFSIZE', None)
    if case.algo != 'Auto':
        os.environ['PCCL_ALGO'] = case.algo
        if re.match(case.algo, 'FC'):
            os.environ['PCCL_DISABLE_FC_GRAPH'] = '0'
    if case.proto != 'Auto':
        os.environ['PCCL_PROTO'] = case.proto
        # Use default buffer sizes, matching the original runner without --mode.


def cmd_rule(args, case):
    """generate cmd_key and cmd according to args and case info."""
    total_nprocs = case.nnodes * case.nprocs
    ranks_pernode = case.ranks // case.nnodes
    ranks_perproc = ranks_pernode // case.nprocs

    g = 1
    ranks_match = True
    MPI = ''
    if case.nprocs * case.nnodes > 1:
        if ranks_pernode % case.nprocs == 0:
            if args.dbg:
                print(
                    "You need to install tmux, openmpi and tmpi in order to debug on mpirun case.")
                MPI = 'tmpi ' + str(case.nprocs) + ' '
            else:
                MPI = " ".join([
                    "mpirun --allow-run-as-root",
                    "-np {}".format(total_nprocs),
                    "-npernode {}".format(case.nprocs),
                ])
            # keep same PCCL_* environment variable on all hosts
            for env_attr, env_val in os.environ.items():
                if re.match("NCCL_", env_attr) or re.match("PCCL_", env_attr):
                    MPI = " ".join([MPI, "-x {}={}".format(env_attr, env_val)])
            MPI = " ".join([MPI, args.mpi_options])
        else:
            ranks_match = False
    if ranks_perproc % case.threads == 0 and ranks_match:
        g = ranks_perproc // case.threads
    else:
        print('ranks does not match with (nnodes * nprocs * threads)')
        sys.exit(-1)
    if args.launch_mode == 'k8s':
        MPI = ''  # non-mpi launch
        case.threads = ranks_pernode  # 1 process per node, 1 ppu per thread
        g = 1  # one GPU per thread in k8s mode

    step = 1 if not args.hw_perf else 2
    data_size_cmd = ""
    if isinstance(case.data_size, list) and len(case.data_size) > 1:
        data_size_cmd = "--sizelist {}".format(','.join([str(x) for x in case.data_size]))
    else:
        if isinstance(case.data_size, list):
            case.data_size = case.data_size[0]
        b_size = obtain_data_size(case.min_data_size) if case.random_size_num > 0 else (
        obtain_data_size(case.data_size) if not args.hw_perf else 8)
        e_size = obtain_data_size(case.max_data_size) if case.random_size_num > 0 else (
        obtain_data_size(case.data_size) if not args.hw_perf else PPUHwPerfCaseDict['HWPerf'][0]['data_size'][-1])
        data_size_cmd = "-b {}".format(b_size) + " -e {}".format(e_size)
        data_size_cmd += " --random_size_num {}".format(case.random_size_num) if case.random_size_num > 0 else " -f {}".format(step)

    # Generate cmd_key after data_size normalization
    # Format data_size for cmd_key: handle list, random, and single size
    if case.random_size_num > 0:
        data_size_str = "{}_{}_{}".format(case.min_data_size, case.max_data_size, case.random_size_num)
    elif isinstance(case.data_size, list):
        data_size_str = ",".join([str(s) for s in case.data_size])
    else:
        data_size_str = str(case.data_size)

    cmd_key = "-".join([
        datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"),
        case.op,
        case.algo,
        case.proto,
        case.redop,
        "{}GPUs".format(case.ranks),
        case.data_type,
        data_size_str,
        "{}Iters".format(case.iters),
        "{}Threads".format(case.threads),
        "{}Procs".format(case.nprocs),
        "{}Nodes".format(case.nnodes)
    ])
    print(cmd_key)

    cmd = " ".join([
        "{}".format(case.op),
        "-t {}".format(case.threads),
        "-g {}".format(g),
        "-d {}".format(case.data_type),
        data_size_cmd,
        "-n {}".format(case.iters),
        "-w {}".format(case.warmup),
        "--core_dump {}".format(args.core_dump),
        "--elapsed_type {}".format(args.elapsed_type),
        "-o {}".format(case.redop),
        "-a {}".format(case.average),
        "-R {}".format(case.register),
        "-x {}".format(case.cta_policy),
        "-p {}".format(case.init_type)
    ])
    if args.hw_perf:
        target_size_list = ','.join(PPUHwPerfCaseDict['HWPerf'][0]['data_size'])
        cmd += ' --sizelist {}'.format(target_size_list)
    # enhance APIs coverage based on current script
    # TODO: refine after arch refactoring
    # single node 8 PPUs and 2 nodes 16 PPUs
    # case filter rule: Comm non Blocking case => Broadcast, Comm Register case => Reduce
    if not args.hw_perf and g * case.threads * case.nprocs == 8 and case.nnodes <= 2:
        if case.op == 'broadcast_perf':
            cmd += ' --init_blocking 0'
        if case.op == 'reduce_perf':
            cmd += ' -R 1'
    if args.no_data_check:
        # disable data_check to keep test pccl kernel is the last launched kernel to eaze dump instrs
        cmd += ' -c 0'
    if args.enable_single_test:
        # enble single test mode to test pccl kernel only on benching period
        cmd += ' -s 1'
    if args.non_blocking:
        cmd += ' --init_blocking 0'
    if args.test_buffer_kind != 2:
        # enble to test pccl kernel by using inplace buffer only or outplace buffer only
        assert args.test_buffer_kind == 0 or args.test_buffer_kind == 1, \
            'Neither inplace nor outplace buffer kind set'
        cmd += ' -k {}'.format(args.test_buffer_kind)
    # enable debug mode on running test
    if args.dbg:
        # enable gdb
        cmd = 'gdb --args ' + cmd
        if args.core_file != '':
            cmd += ' ' + args.core_file
    # finalize command before issuing tests
    cmd = " ".join([MPI, cmd])
    if args.profile == 'acu':
        cmd = 'acu --set full -f -o {} --target-processes all --replay-mode {} '.format(
            cmd_key, args.replay_mode) + cmd + " --acu_profile 1 -n {} --target_process {}".format(args.iters[0], args.target_proc)
    print(cmd)
    return cmd_key, cmd


def add_ignore_params(parser):
    """add ignore params into seprated group."""

    group = parser.add_argument_group(title='ignore list params group')

    group.add_argument('--ignore-op-list', nargs="*", default=[], type=str, metavar='',
                       help='ignore op in this list')
    group.add_argument('--ignore-algo-list', nargs="*", default=[], type=str, metavar='',
                       help='ignore algo in this list')
    group.add_argument('--ignore-proto-list', nargs="*", default=[], type=str, metavar='',
                       help='ignore proto in this list')
    group.add_argument('--ignore-ranks-list', nargs="*", default=[], type=int, metavar='',
                       help='ignore ranks in this list')
    group.add_argument('--ignore-data-size-list', nargs="*", default=[], type=str, metavar='',
                       help='ignore data size in this list')
    group.add_argument('--ignore-data-type-list', nargs="*", default=[], type=str, metavar='',
                       help='ignore data type in this list')
    group.add_argument('--ignore-iters-list', nargs="*", default=[], type=int, metavar='',
                       help='ignore iters in this list')
    group.add_argument('--ignore-threads-list', nargs="*", default=[], type=int, metavar='',
                       help='ignore threads in this list')
    group.add_argument('--ignore-nprocs-list', nargs="*", default=[], type=int, metavar='',
                       help='ignore nprocs in this list')
    group.add_argument('--ignore-nnodes-list', nargs="*", default=[], type=int, metavar='',
                       help='ignore nnodes in this list')


def obtain_test_params():
    """parse command parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', nargs="*", default=['all_gather_perf', 'all_reduce_perf',
                        'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf', 'alltoall_perf',
                                                    'gather_perf', 'scatter_perf', 'sendrecv_perf', 'hypercube_perf'],
                        type=str, metavar='', help='select colloective op list')
    parser.add_argument('--algo', nargs="*", default=['Ring'], type=str, metavar='',
                        help='choose which algorithm list to run')
    parser.add_argument('--proto', nargs="*", default=['LL'], type=str, metavar='',
                        help='choose which proto to run')
    parser.add_argument('--redop', nargs="*", default=['sum'],
                        type=str, metavar='', help='select colloective reduce op list')
    parser.add_argument('--ranks', nargs="*", default=[4], type=int, metavar='',
                        help='total ranks which equals to pcclComm.nRanks')
    parser.add_argument('--data-type', nargs="*", default=['float'], type=str, metavar='',
                        help='choose data type')
    parser.add_argument('--data-size', nargs="*", default=['32B'], type=str, metavar='',
                        help='choose data size')
    parser.add_argument('--min-data-size', nargs="*", default=['32B'], type=str, metavar='',
                        help='choose min data size')
    parser.add_argument('--max-data-size', nargs="*", default=['32MB'], type=str, metavar='',
                        help='choose max data size')
    parser.add_argument('--random-size-num', default=0, type=int, metavar='',
                        help='generate random size between min and max data size')
    parser.add_argument('--iters', nargs="*", default=[5], type=int, metavar='',
                        help='choose test iterations')
    parser.add_argument('--file', default='result.txt', type=str, metavar='',
                        help='choose test iterations')
    parser.add_argument('--timeout', default=7200, type=int, metavar='',
                        help='configure timeout time seconds for each case')
    parser.add_argument('--core-dump', default=0, type=int, metavar='',
                        help='configure if enable core dump test')
    parser.add_argument('--core-file', default='', type=str, metavar='',
                        help="configure core dump files used to do debug analysis")
    parser.add_argument('--case-dump', default=0, type=int, metavar='',
                        help='configure if dump test case only')
    parser.add_argument('--threads', nargs="*", default=[1], type=int, metavar='',
                        help='configure thread numbers per process')
    parser.add_argument('--all', default=0, type=int, metavar='',
                        help='configure if run full cases')
    parser.add_argument('--check-result', default=0, type=int, metavar='',
                        help='choose test failed case again from file')
    parser.add_argument('--suppress-output', default=0, type=int, metavar='',
                        help='configure if suppress test command log output')
    parser.add_argument('--dbg', action='store_true',
                        help='use gdb to debug current case')
    parser.add_argument('--stop-once-failed', action='store_true',
                        help='stop test immediately if met any error')
    parser.add_argument('--no-data-check', action='store_true',
                        help='disable pccl-test case data check phase')
    parser.add_argument('--enable-single-test', action='store_true',
                        help='enable pccl-test case benchmark test only')
    parser.add_argument('--test-buffer-kind', default=2, type=int, metavar='',
                        help='configure which kind of buffer \
                        used to do collective test like inplace/outplace/inOutplace by setting 0/1/2 separately')
    parser.add_argument('--group-launch', action='store_true',
                        help='use group launch mode to launch pccl coll kernels')
    parser.add_argument('--nprocs', nargs="*", default=[1], type=int, metavar='',
                        help='configure process numbers per node')
    parser.add_argument('--nnodes', nargs="*", default=[1], type=int, metavar='',
                        help='configure total host nodes')
    parser.add_argument('--mpi-options', default='', type=str, metavar='',
                        help="mpirun options. e.g. --hostfile <filepath>, -mca btl_tcp_if_include <ip name>, -x <ENV>, ...")
    parser.add_argument('--profile', choices=['acu', 'asys', 'null'], default='null', type=str, metavar='',
                        help="configure if enable profile mode, and the detail type, could be 'acu' or 'asys'")
    parser.add_argument('--replay-mode', choices=['range', 'app-range'], default='range', type=str, metavar='',
                        help="choose the detail replay mode, could be 'range' or 'app-range' ")
    parser.add_argument('--target-proc', default=-1, type=int, metavar='',
                        help="for acu app-range, choose to profile target process")
    parser.add_argument('--elapsed_type', default=0, type=int, metavar='',
                        help="choose to count elapsed time by host clock or hggc event"
                        " default is 0 means host clock")
    parser.add_argument('--cta-policy', default=0, type=int, metavar='',
                        help='choose cta policy type')
    parser.add_argument('--warmup', default=0, type=int, metavar='')
    parser.add_argument('--average', default=2, type=int, metavar='')
    parser.add_argument('--auto-detect', action='store_true',
                        help='use auto detection way to launch pccl coll kernels')
    parser.add_argument('--register', default=0, type=int, choices=[0, 1, 2],
                        help='register sendbuff and recvbuff')
    parser.add_argument('--devices', default='', type=str, metavar='',
                        help='choose some devices')
    parser.add_argument('--hw-perf', action='store_true',
                        help='configure if run ppu perf scan case.')
    parser.add_argument('--cross-node', choices=['gdr', 'non_gdr', 'socket', 'icn_scale_out'], default='socket', type=str, metavar='',
                        help='configure cross node mode.')
    parser.add_argument('--launch-mode', choices=['normal', 'k8s'], default='normal', type=str, metavar='',
                        help='configure pccl test process launch mode.')
    parser.add_argument('--use_4nics', action='store_true',
                        help='configure if run case under 4 NICs mode')
    parser.add_argument('--non-blocking', action='store_true',
                        help='configure if run pccl non-blocking mode')
    parser.add_argument('--process-group', choices=['normal', 'isolation'], default='isolation', type=str, metavar='',
                        help='configure pccl test process group mode.')
    parser.add_argument('--init_type', default=0, type=int, choices=[0, 1, 2],
                        help='0:config, 1:parallel, 2:scalable')

    add_ignore_params(parser)

    args = parser.parse_args()
    return args


def fill_case_list_from_args(args):
    """generate case list from args."""
    CaseList.clear()
    case_dict = TestCase.get_case_dict_from_args(args)
    data_size_list = case_dict['data_size']
    case_dict['data_size'] = ['']
    TestCase.gen_product_zip_cases(case_dict, CaseList, args, [
                                   ['min_data_size', 'max_data_size']])
    for idx in range(len(CaseList)):
        setattr(CaseList[idx], 'data_size', data_size_list)


def fill_case_list_from_case_dict(args, CaseDicts, zip_groups=[[]]):
    """generate case list from CaseMapList."""
    CaseList.clear()
    for case_name in CaseDicts.keys():
        for case_dict in CaseDicts[case_name]:
            TestCase.gen_product_zip_cases(case_dict, CaseList, args, zip_groups=zip_groups)

def fill_case_list_from_txt(args):
    """generate case list from result.txt """
    CaseList.clear()
    with open(args.file, 'r') as result_file:
        lines = result_file.readlines()
        for line in lines:
            cmd_line = line.split()[0]
            _result = line.split()[2]
            _, _, _op, _algo, _proto, redop, _ranks, _data_type, _data_size, _iters, _threads, _procs, _nodes = cmd_line.split(
                '-')
            _ranks = int(re.sub(r"\D", "", _ranks))
            _iters = int(re.sub(r"\D", "", _iters))
            _threads = int(re.sub(r"\D", "", _threads))
            _procs = int(re.sub(r"\D", "", _procs))
            _nodes = int(re.sub(r"\D", "", _nodes))

            # Parse data_size field: could be single size, multi-size list (comma-separated), or random size range (underscore-separated)
            _min_data_size = None
            _max_data_size = None
            _random_size_num = 0

            # Check for random size format first: min_max_count (with underscores)
            if '_' in _data_size:
                parts = _data_size.split('_')
                if len(parts) == 3 and parts[2].isdigit():
                    # Random size format: 32B_32MB_3
                    _min_data_size = parts[0]
                    _max_data_size = parts[1]
                    _random_size_num = int(parts[2])
                    _data_size = None
                else:
                    # Old format with underscores (should not happen with new code, but keep for compatibility)
                    _data_size = parts

            # Check for multi-size format: comma-separated list
            if _data_size and ',' in _data_size:
                _data_size = _data_size.split(',')

            if _result == 'FAIL':
                if _random_size_num > 0:
                    CaseList.append(TestCase(args, op=_op, algo=_algo, proto=_proto, redop=redop, ranks=_ranks, data_type=_data_type,
                                    data_size=_data_size, min_data_size=_min_data_size, max_data_size=_max_data_size,
                                    random_size_num=_random_size_num, iters=_iters, threads=_threads, nprocs=_procs, nnodes=_nodes))
                else:
                    CaseList.append(TestCase(args, op=_op, algo=_algo, proto=_proto, redop=redop, ranks=_ranks, data_type=_data_type,
                                    data_size=_data_size, iters=_iters, threads=_threads, nprocs=_procs, nnodes=_nodes))


def fill_case_list_from_list(args, cmd_str):
    """generate case list fram list."""
    cmd_str_list = cmd_str.split('_')
    op_idx = cmd_str_list.index('op')
    algo_idx = cmd_str_list.index('algo')
    proto_idx = cmd_str_list.index('proto')
    ranks_idx = cmd_str_list.index('ranks')
    data_type_idx = cmd_str_list.index('datatype')
    data_size_idx = cmd_str_list.index('datasize')
    iters_idx = cmd_str_list.index('iters')
    threads_idx = cmd_str_list.index('threads')

    _op = ''
    for index in range(op_idx + 1, algo_idx):
        if _op != '':
            _op += '_'
        _op = _op + cmd_str_list[index]
    CaseList.append(TestCase(args, op=_op, algo=cmd_str_list[algo_idx + 1], proto=cmd_str_list[proto_idx + 1], ranks=int(cmd_str_list[ranks_idx + 1]),
                             data_type=cmd_str_list[data_type_idx + 1], data_size=cmd_str_list[data_size_idx + 1], iters=int(
                                 cmd_str_list[iters_idx + 1]),
                             threads=int(cmd_str_list[threads_idx + 1]), nprocs=1, nnodes=1))  # TODO: fixed nprocs


def fill_case_list_from_xls(args):
    """obtain case list from excel xls file."""
    import xlrd
    CaseList.clear()
    work_book = xlrd.open_workbook(args.file)
    sheet = work_book.sheet_by_index(0)
    sheet.cell_value(0, 0)
    for i in range(1, sheet.nrows):
        # print(sheet.cell_value(i, 0))
        fill_case_list_from_list(args, sheet.cell_value(i, 0))


def fill_case_list_from_xlsx(args):
    """obtain case list from excel xlsx file."""
    import openpyxl
    CaseList.clear()
    work_book = openpyxl.load_workbook(args.file)
    sheet = work_book.active
    for column in sheet.iter_cols(1, 1):
        for row in range(1, sheet.max_row):
            # print(column[row].value)
            fill_case_list_from_list(args, column[row].value)


def fill_case_list_from_dump_list_file(args, list_file):
    """obtain daily test cases."""
    with open(list_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            cmd_list = line.split()
            _op = cmd_list[cmd_list.index('--op') + 1]
            _algo = cmd_list[cmd_list.index('--algo') + 1]
            _proto = cmd_list[cmd_list.index('--proto') + 1]
            _ranks = int(cmd_list[cmd_list.index('--ranks') + 1])
            _data_type = cmd_list[cmd_list.index('--data-type') + 1]
            _data_size = cmd_list[cmd_list.index('--data-size') + 1]
            _iters = int(cmd_list[cmd_list.index('--iters') + 1])
            _threads = int(cmd_list[cmd_list.index('--threads') + 1])
            CaseList.append(TestCase(args, op=_op, algo=_algo, proto=_proto,
                                     ranks=_ranks, data_type=_data_type, data_size=_data_size,
                                     iters=_iters, threads=_threads, nprocs=1, nnodes=1))


def fill_case_list_from_result(args):
    """generate case list from txt/xls/xlsx/list file."""
    _, file_extension = os.path.splitext(args.file)
    if file_extension == '.txt':
        fill_case_list_from_txt(args)
    elif file_extension == '.xls':
        fill_case_list_from_xls(args)
    elif file_extension == '.xlsx':
        fill_case_list_from_xlsx(args)
    elif file_extension == '.list':
        fill_case_list_from_dump_list_file(args, args.file)
    else:
        assert False

    # rename result file to avoid conflict.
    args.file = 'check_result.txt'
    try:
        os.remove(args.file)
    except OSError:
        pass


def params_check():
    """check input parameters."""
    assert set(args.op).issubset(set(CollOps)), CollOps
    assert set(args.algo).issubset(set(CollAlgos)), CollAlgos
    assert set(args.proto).issubset(set(CollProtos)), CollProtos
    assert set(args.data_type).issubset(set(CollDataTypes)), CollDataTypes


def dump_json(file_name, case_list, hw_type, gen_name_args_fn, **fn_args):
    """Generate corresponding case information according to parameters and gen_name_args_fn, and dump it into json format."""
    import json
    case_entry = "pccl-tests/run_test.py"
    case_run_type = "python"
    cases_json = []
    with open(file_name, 'w') as case_json_file:
        for case in case_list:
            new_case_name_list, new_case_args_list = gen_name_args_fn(
                case, **fn_args)
            # begin construct case list
            for case_name, case_args in zip(new_case_name_list, new_case_args_list):
                case_dict = {}
                case_dict["entry"] = case_entry
                case_dict["casename"] = case_name
                case_dict["run_type"] = case_run_type
                case_dict["args"] = case_args
                case_dict["hw_type"] = hw_type
                cases_json.append(case_dict)
        json.dump(cases_json, case_json_file, indent=4)


def gen_py_case_name_args(case):
    """gen basic case name and args"""
    case_name = "op_{}_algo_{}_proto_{}_redop_{}_ranks_{}_datatype_{}_datasize_{}_iters_{}_threads_{}_nprocs_{}_nnodes{}" \
        .format(case.op, case.algo, case.proto, case.redop, case.ranks, case.data_type, case.data_size, case.iters, case.threads, case.nprocs, case.nnodes)

    case_arg = " ".join(["--op {}".format(case.op), "--algo {}".format(case.algo), "--proto {}".format(case.proto), "--redop {}".format(case.redop),
                         "--ranks {}".format(case.ranks), "--data-type {}".format(
                             case.data_type), "--data-size {}".format(case.data_size), "--iters {}".format(case.iters),
                         "--threads {}".format(case.threads), "--nprocs {}".format(
                             case.nprocs), "--nnodes {}".format(case.nnodes)
                         ])
    return case_name, case_arg


def gen_py_crossnode_name_args_list(case, crossnode):
    """gen crossnode case's name and args"""
    case_name_list = []
    case_arg_list = []
    case_name, case_arg = gen_py_case_name_args(case)
    case_name = case_name + '_autodetect'
    case_arg = case_arg + ' --auto-detect'
    if crossnode == 2:
        if case.proto == 'Signal':
            return case_name_list, case_arg_list
        rdma_case_args = case_arg + " --launch-mode k8s --cross-node"
        # potential data corruption due to network or PCI-E switch package split and relax reorder behavior,
        # which will not guarantee 128-byte atmoic.
        if case.proto != 'LL128':
            case_name_list.append(case_name + '_rdma_gdr')
            case_arg_list.append(" ".join([rdma_case_args, "gdr"]))
            if case.ranks / case.nnodes == 8 and case.op in ['all_reduce_perf', 'alltoall_perf'] and case.data_type == 'half':
                case_name_list.append(case_name + '_rdma_gdr_4nics')
                case_arg_list.append(
                    " ".join([rdma_case_args, "gdr --use_4nics"]))

                case_name_list.append(case_name + '_rdma_host')
                case_arg_list.append(" ".join([rdma_case_args, "non_gdr"]))
                case_name_list.append(case_name + '_socket')
                case_arg_list.append(" ".join([rdma_case_args, "socket"]))

        elif case.nnodes == 2 and case.ranks == 16:
            case_name_list.append(case_name + '_rdma_gdr')
            case_arg_list.append(" ".join([rdma_case_args, "gdr"]))

    elif crossnode == 1:
        case_name_list.append(case_name + "")
        case_arg_list.append(" ".join(
            [case_arg, "--mpi-options='-mca btl_tcp_if_include <IFNAME> -x PCCL_ICN_SCALE_OUT_DISABLE=0 --hostfile ./hosts.cfg'"]))
    return case_name_list, case_arg_list


def case_dump_ignore(case, args, **black_list):
    """ignore some cases when case dump."""
    if not TestCase.is_valid(case, args):
        return True
    for key, val in black_list.items():
        if getattr(case, key, None) in val:
            return True
    return False


def case_dump(args):
    """dump test case."""
    def _gen_case_name_args_list(case, **kwargs):
        multi_process = kwargs['multi_process']
        crossnode = kwargs['crossnode']
        mpi_split_mask_and_average = kwargs['mpi_split_mask_and_average']
        case_name_list, case_args_list = [], []
        _ranks_pernode = case.ranks // case.nnodes
        basic_case_name, basic_case_args = gen_py_case_name_args(case)
        if mpi_split_mask_and_average:
            if case.threads * case.nnodes * case.nprocs != case.ranks or case.ranks / case.nnodes > 16:
                return case_name_list, case_args_list
            if TestCase.is_subset(case, CaseDumpSplitMaskAverage['AverageDict']):
                for _average in ConfigDict['CaseDump']['average']:
                    case_name = basic_case_name + \
                        "_average{}_autodetect".format(_average)
                    case_args = basic_case_args + \
                        " --average {} --auto-detect".format(_average)
                    case_name_list.append(case_name)
                    case_args_list.append(case_args)

            if TestCase.is_subset(case, CaseDumpSplitMaskAverage['MaskDict']):
                for _mask in ConfigDict['CaseDump']['mask']:
                    case_name = basic_case_name + \
                        "_mask{}_autodetect".format(_mask)
                    case_args = basic_case_args + \
                        " --mpi-options='-mca btl_tcp_if_include <IFNAME> -x NCCL_TESTS_SPLIT_MASK={} --hostfile ./hosts.cfg' --auto-detect".format(
                            _mask)
                    case_name_list.append(case_name)
                    case_args_list.append(case_args)
        elif not multi_process:
            # single node single process
            if case.nnodes != 1 or case.nprocs != 1 or (case.threads != 1 and case.threads != 2) or case.ranks > 8:
                return case_name_list, case_args_list
            case_name_list.append(basic_case_name)
            case_args_list.append(basic_case_args)
        elif not crossnode:
            # single node multiple processes
            if case.nnodes > 1 or case.nprocs != case.ranks or case.threads != 1 or case.ranks > 16:
                return case_name_list, case_args_list
            case_name_list.append(basic_case_name + "_autodetect")
            case_args_list.append(basic_case_args + " --auto-detect")
        else:
            # multiple nodes multiple processes
            if case.nnodes <= 1 or case.nnodes * case.nprocs != case.ranks or case.threads != 1:
                return case_name_list, case_args_list
            name_list, args_list = gen_py_crossnode_name_args_list(
                case, crossnode)
            case_name_list = case_name_list + name_list
            case_args_list = case_args_list + args_list
        return case_name_list, case_args_list

    new_case_list = [case for case in CaseList if not case_dump_ignore(
        case, args, data_size=['8MB', '32MB', '256MB'], iters=[1, 10])]
    full_case_list = [case for case in CaseList]
    with open('pccl-tests.list', 'w') as case_list_file:
        for case in new_case_list:
            if case.threads != 1 and case.threads != 2:
                continue
            _, basic_case_args = gen_py_case_name_args(case)
            if case.nnodes > 1:
                case_args = "python -u ./run_test.py"
            else:
                case_args = "python ./run_test.py"
            case_args = " ".join([case_args, basic_case_args])
            case_list_file.write(case_args)
            case_list_file.write('\n')



    # file_name, multi_process, crossnode, case_list, mpi_split_mask_and_average, hw_type
    config_map = {
        'single_process': ['pccl-tests.json', False, 0, new_case_list, 0, ["ppu"]],
        'multi_process': ['pccl-tests-multi-process.json', True, 0, new_case_list, 0, ["ppu"]],
        'icn_crossnode': ['pccl-tests-icn-crossnode.json', True, 1, full_case_list, 0, ["ppu"]],
        'net_crossnode': ['pccl-tests-net-crossnode.json', True, 2, full_case_list, 0, ["ppu"]],
        'mpi_split_mask_and_average': ['pccl-tests-mpi-split-mask-and-average.json', False, 0, full_case_list, 1, ["ppu"]]
    }
    for key in config_map.keys():
        file_name, multi_process, crossnode, case_list, mpi_split_mask_and_average, hw_type = config_map[
            key]
        dump_json(file_name, case_list, hw_type, _gen_case_name_args_list,
                  multi_process=multi_process, crossnode=crossnode, mpi_split_mask_and_average=mpi_split_mask_and_average)


def ppu_case_dump(args, caseDictTag, configDictName = None):
    """dump test case."""
    def _gen_auto_detect_args_list(case, **kwargs):
        case_name_list, case_args_list = [], []
        case_name = "op_{}_algo_{}_proto_{}_datatype_{}_datasize_{}_cta_policy_{}_iters_{}" \
            .format(case.op, case.algo, case.proto, case.data_type, case.data_size, case.cta_policy, case.iters)
        case_args = "--op {} --algo {} --proto {} --data-type {} --data-size {} --cta-policy {} --iters {}" \
            .format(case.op, case.algo, case.proto, case.data_type, case.data_size, case.cta_policy, case.iters)
        if configDictName:
            devices_list=configDictName['PPUCaseDump']['devices_list']
            nproc_threads_list=configDictName['PPUCaseDump']['nproc_threads_list']
            for _devices_list in devices_list:
                case_name_list.append(case_name + "_ranks_{}".format(_devices_list.count(
                    ",") + 1) + "_devices_{}".format(_devices_list.replace(",", "_")))
                case_args_list.append(case_args + " --ranks {}".format(_devices_list.count(
                    ",") + 1) + " --devices {}".format(_devices_list) + " --auto-detect")
            for _nproc_threads_list in nproc_threads_list:
                case_name_list.append(case_name + "_ranks_8" + "_nproc_{}".format(
                    _nproc_threads_list[0]) + "_threads_{}".format(_nproc_threads_list[1]))
                case_args_list.append(case_args + " --ranks 8" + " --nproc {}".format(
                    _nproc_threads_list[0]) + " --threads {}".format(_nproc_threads_list[1]) + " --auto-detect")
        else:
            case_args_list.append(case_args + " --ranks {}".format(case.ranks))
            case_name_list.append(case_name + "_ranks_{}".format(case.ranks))
        return case_name_list, case_args_list
    new_case_list = [
        case for case in CaseList if not case_dump_ignore(case, args)]
    with open('pccl-tests.ppu.list', 'w') as case_list_file:
        for case in new_case_list:
            # considering different topology types
            _, case_args_list = _gen_auto_detect_args_list(
                case)
            for new_case_name in case_args_list:
                case_list_file.write("python ./run_test.py " + new_case_name)
                case_list_file.write('\n')
    file_name = 'pccl-tests-' + caseDictTag + '.ppu.json'
    hw_type = ['ppu']
    dump_json(file_name, new_case_list, hw_type, gen_name_args_fn=_gen_auto_detect_args_list)


def ppu_hw_perf_case_dump(args):
    """dump ppu hw perf test case."""
    def _gen_ppu_perf_case_list(case):
        case_name_list, case_args_list = [], []
        basic_name, basic_args = gen_py_case_name_args(case)

        case_name = "ppu_perf_" + basic_name + "_autodetect"
        case_args = basic_args + " --warmup 5 --auto-detect --hw-perf --elapsed_type 1"
        if case.nnodes > 1:
            case_args = ' '.join(
                [case_args, '--launch-mode k8s --cross-node gdr'])
            if case.op in ['all_reduce_perf', 'alltoall_perf']:
                case_name_list.append(case_name + '_4nics')
                case_args_list.append(case_args + ' --use_4nics')
        case_name_list.append(case_name)
        case_args_list.append(case_args)

        return case_name_list, case_args_list

    import json
    case_list = [
        case for case in CaseList if not case_dump_ignore(case, args)]
    case_entry = "pccl-tests/run_test.py"
    case_run_type = "python"
    case_names = []
    cases_json_8ppus = []
    cases_json_16ppus = []
    file_name_8ppus = open('pccl-tests-8ppus-per-node-perf.json', 'w')
    file_name_16ppus = open('pccl-tests-16ppus-per-node-perf.json', 'w')

    for case in case_list:
        ppus_per_node = case.ranks // case.nnodes
        new_case_name_list, new_case_args_list = _gen_ppu_perf_case_list(case)

        # begin construct case list
        for case_name, case_args in zip(new_case_name_list, new_case_args_list):
            if case_name in case_names:
                raise RuntimeError(
                    f"do not allow repeat casename, {case_name}")
            case_names.append(case_name)
            case_dict = {}
            case_dict["entry"] = case_entry
            case_dict["casename"] = case_name
            case_dict["run_type"] = case_run_type
            case_dict["args"] = case_args
            case_dict["hw_type"] = ["ppu"]
            cases_json_16ppus.append(case_dict)
            if ppus_per_node <= 8:
                cases_json_8ppus.append(case_dict)

    json.dump(cases_json_8ppus, file_name_8ppus, indent=4)
    json.dump(cases_json_16ppus, file_name_16ppus, indent=4)


def ppu_hw_perf_process(case, output, args):
    case_name = "{}_{}_{}_ranks_{}_nnodes_{}" \
        .format(case.op, case.algo, case.proto, case.ranks, case.nnodes)
    if args.use_4nics:
        case_name = case_name + '_4nics'

    index = []  # store out-of-place and in-place time index

    def get_index(op):
        if op == 'alltoall_perf':
            # out-of-place time
            return index[0]
        else:
            # inplace time
            return index[1]

    with open('ppu_hw_perf.report', 'a+') as f:
        for line in output.splitlines():
            if 'time' in line and 'busbw' in line:
                index = [idx - 1 for idx, value in enumerate(
                    line.split()) if value in ('cpu_time', 'event_time', 'cupti_time')]
            if case.data_type in line:
                index_base = get_index(case.op)
                size = line.split()[0]
                f.write(case_name + '_' + str(size) +
                        ':' + line.split()[index_base] +
                        ':' + line.split()[index_base + 1] + '\n')


def obtain_data_size(str_data_size):
    """calculate data size from string value."""
    unit = str_data_size.translate(str.maketrans('', '', string.digits))
    num = str_data_size.translate(str.maketrans('', '', string.ascii_letters))
    coef = 1
    if 'K' in unit:
        coef = 1024
    elif 'M' in unit:
        coef = 1024 * 1024
    elif 'G' in unit:
        coef = 1024 * 1024 * 1024

    return int(coef * float(num))


def get_string_from_exit_code(code=0):
    """fomat string from command exit code."""
    result_str = 'FAIL'
    if code == 0:
        result_str = 'PASS'
    elif code == 1:  # signal Abort
        result_str = result_str + " (Test Case Failed)"
    elif code == 2:  # pccl-tests HGGC call check failed
        result_str = result_str + " (HGGC Call Check Failed)"
    elif code == 3:  # pccl-tests PCCL call check failed
        result_str = result_str + " (PCCL Call Check Failed)"
    elif code == -6:  # signal 6
        result_str = result_str + " (Test Command Aborted)"
    elif code == -9:  # signal 9
        result_str = result_str + " (Test Command Killed)"
    elif code == -11:  # signal 11
        result_str = result_str + " (Test Command Segmentation fault)"
    elif code == -15:  # signal Abort
        result_str = result_str + " (Test Command Timeout)"
    else:
        result_str = result_str + " (Unknown Exit Code : " + str(code) + ")"

    return result_str


def clean_exit(signum, frame):
    """kill all processes in my group."""
    os.killpg(0, signal.SIGKILL)
    sys.exit()


def store_coredump(cmd_info, ret_code):
    """reanme current core dump file with command and arguments info, also time."""
    # if disable core_dump, just skip for it.
    if args.core_dump == 0:
        return
    if ret_code == 0 or ret_code == 1:
        print("no core dump for them")
    else:
        # for cmd crash or abort, it will trigger core dump and generate "core" file
        # rename core dump file to "cmd-args-signal.core" if has
        try:
            os.rename("core", cmd_info + "-signal" + str(ret_code) + "-" +
                      datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".core")
            # force to exit due to each core dump file is nearly 7GB.
            sys.exit()
        except OSError:
            pass

# Print iterations progress


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=200, fill='=', printEnd="\r"):
    """Call in a loop to create terminal progress bar.

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    # percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    percent = str(iteration) + "/" + str(total)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    # print(f'\r{prefix} [{bar}] {percent}% {suffix}', end=printEnd)
    # print(f'\r{prefix} [{bar}] {percent} {suffix}', end=printEnd)
    print('\r', prefix, ' [', bar, '] ', percent, ' ', suffix, end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    args = obtain_test_params()

    if args.process_group != 'normal':
        # create new process group, become its leader
        os.setpgrp()
        # captrue Ctrl+c and term
        signal.signal(signal.SIGINT, clean_exit)
        signal.signal(signal.SIGTERM, clean_exit)

    if args.all == 1:
        # obtain case list from CaseMap
        fill_case_list_from_case_dict(args, CommonCaseDict)
    elif args.check_result == 1:
        # obtain case list from result.txt
        fill_case_list_from_result(args)
    elif args.case_dump == 1:
        # obtain case list from CaseMap
        fill_case_list_from_case_dict(args, CommonCaseDict)
        case_dump(args)
        sys.exit()
    elif args.case_dump == 2:
        # obtain case list from PPUCaseMap
        fill_case_list_from_case_dict(args, PPUCaseDict)
        ppu_case_dump(args, "perf", ConfigDict)
        sys.exit()
    elif args.case_dump == 3:
        # obtain case list from PPUHwPerfCaseMap
        fill_case_list_from_case_dict(args, PPUHwPerfCaseDict, zip_groups=[['ranks', 'nnodes']])
        ppu_hw_perf_case_dump(args)
        sys.exit()
    elif args.case_dump == 4:
        fill_case_list_from_case_dict(args, CtaPolicyCaseSM80Dict)
        ppu_case_dump(args, "cta-policy-sm80")
        sys.exit()
    elif args.case_dump == 5:
        fill_case_list_from_case_dict(args, CtaPolicyCaseSM89Dict)
        ppu_case_dump(args, "cta-policy-sm89")
        sys.exit()
    else:
        params_check()
        # obtain case list from args
        fill_case_list_from_args(args)

    if len(CaseList) <= 0:
        print("case list is empty.")
        sys.exit()

    result_list = []
    total_cases = 0
    pass_cases = 0

    # Initial call to print 0% progress
    printProgressBar(0, len(CaseList), prefix='Progress:', suffix='Complete')
    import filelock
    apply_env_args_rule(args)
    for case in CaseList:
        apply_env_case_rule(args, case)
        cmd_key, cmd = cmd_rule(args, case)
        exit_code = 0
        # disable timeout
        timeout = args.timeout
        if timeout <= 0 or args.dbg:
            timeout = None
        try:
            cmd_process = None
            if args.hw_perf:
                cmd_process = subprocess.run(cmd.strip(), capture_output=True,
                                             encoding='utf-8', timeout=timeout, shell=True, check=True)
            else:
                # Support keeping spaces within quoted strings
                if args.suppress_output:
                    cmd_process = subprocess.run(shlex.split(cmd), timeout=timeout, check=True,
                                                 stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.STDOUT)
                else:
                    cmd_process = subprocess.run(
                        shlex.split(cmd), timeout=timeout, check=True)
            exit_code = cmd_process.returncode
        except subprocess.TimeoutExpired:
            exit_code = -15
        except subprocess.CalledProcessError as ex:
            exit_code = ex.returncode
        except Exception as Err:
            print('unexpected execption: ', Err)
            exit_code = -100  # customized code for debug

        result = get_string_from_exit_code(exit_code)
        if result == 'PASS':
            pass_cases = pass_cases + 1
            result_list.append("\033[92m " + cmd_key +
                               " : " + result + "\033[00m")
            if args.hw_perf:
                ppu_hw_perf_process(case, cmd_process.stdout, args)
        else:
            result_list.append("\033[91m " + cmd_key +
                               " : " + result + "\033[00m")
        total_cases += 1

        # store result for each case
        lock = filelock.FileLock('.' + args.file + '.lock', timeout=10)
        with lock:
            open(args.file, 'a+').write(cmd_key + " : " + result + "\n")

        # store core dump file if need
        store_coredump(cmd_key, exit_code)

        # Update Progress Bar
        printProgressBar(total_cases, len(CaseList),
                         prefix='Progress:', suffix='Complete')

        if args.stop_once_failed and exit_code != 0:
            print(
                "\033[91m hit failed case: {}, and stop test !!!\033[00m".format(cmd_key))
            sys.exit(-1)
    # print result list
    for val in result_list:
        print(val)
    print("\n")
    print("\033[93m " + "Total Cases: " + str(total_cases),
          ", Fail: ", str(total_cases - pass_cases),
          ", Pass: ", str(pass_cases),
          ", PassRate: ", str(100 * pass_cases / total_cases if total_cases != 0 else 0.), "% \033[00m \n")

    # return with fail case num
    sys.exit(total_cases - pass_cases)
