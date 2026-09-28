CollOps = ['all_gather_perf', 'all_reduce_perf', 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf',
           'alltoall_perf', 'gather_perf', 'scatter_perf', 'sendrecv_perf', 'hypercube_perf']
CollAlgos = ['Tree', 'Ring', 'FC', 'Auto']
CollProtos = ['LL', 'LL128', 'Simple', 'Signal',
              'SimpleBulk', 'SignalBulk', 'Auto']
CollDataTypes = ['int8', 'uint8', 'int32', 'uint32',
                 'int64', 'uint64', 'half', 'float', 'double', 'bf16', 'f8e4m3', 'f8e5m2']

"""

CaseDicts ={
        'name':[case_dict1, case_dict2,....]
}

"""

CommonCaseDict = {
    'CommonCases':
        [
            {
                'op': ['alltoall_perf', 'gather_perf', 'scatter_perf', 'sendrecv_perf', 'hypercube_perf'],
                'algo': ['Ring'],
                'proto': ['Auto'],
                'redop': ['sum'],
                'ranks': [2, 4, 8, 16, 32],
                'data_type': ['int8'],
                'data_size': ['32B', '32KB', '8MB', '32MB', '256MB'],
                'iters': [10],
                'threads': [1, 2, 4, 8],
                'nprocs': [1, 2, 4, 8],
                'nnodes': [1, 2, 4]
            },
            {
                'op': ['all_gather_perf', 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf'],
                'algo': ['Ring'],
                'proto': ['LL', 'LL128', 'SimpleBulk'],
                'redop': ['sum'],
                'ranks': [2, 4, 8, 16, 32],
                'data_type': ['int8'],
                'data_size': ['32B', '32KB', '8MB', '32MB', '256MB'],
                'iters': [10],
                'threads': [1, 2, 4, 8],
                'nprocs': [1, 2, 4, 8],
                'nnodes': [1, 2, 4]
            },
            {
                'op': ['all_reduce_perf'],
                'algo': ['Ring', 'Tree'],
                'proto': ['LL', 'LL128', 'Simple', 'SimpleBulk', 'Signal', 'SignalBulk'],
                'redop': ['sum', 'avg'],
                'ranks': [2, 4, 8, 16, 32],
                'data_type': ['int8', 'half', 'float', 'double', 'bf16'],
                'data_size': ['32B', '32KB', '8MB', '32MB', '256MB'],
                'iters': [10],
                'threads': [1, 2, 4, 8],
                'nprocs': [1, 2, 4, 8],
                'nnodes': [1, 2, 4]
            },
        ],
    # for Resnet50 payload check.
    'Resnet50Cases': [
            {
                'op': ['all_reduce_perf'],
                'algo': ['Auto'],
                'proto': ['Auto'],
                'ranks': [2, 4],
                'data_type': ['float'],
                'data_size': ['2236736', '1186344'],
                'iters': [2],
                'threads': [1],
                'nprocs': [1, 2, 4, 8],
                'nnodes': [1, 2]
            },
            {
                'op': ['broadcast_perf'],
                'algo': ['Auto'],
                'proto': ['Auto'],
                'ranks': [2, 4],
                'data_type': ['int8'],
                'data_size': ['3440104'],
                'iters': [2],
                'threads': [1],
                'nprocs': [1, 2, 4, 8],
                'nnodes': [1, 2]
            }],
    # special case
    'SpecialCases': [{
        'op': ['all_reduce_perf'],
        'algo': ['Ring'],
        'proto': ['LL'],
        'ranks': [2],
        'data_type': ['double'],
        'data_size': ['1KB'],
        'iters': [2],
        'threads': [1],
        'nprocs': [1, 2, 4, 8],
        'nnodes': [1, 2]
    }],
    # corner case map, model test NaN issue collective op config
    'CornerCases': [{
        'op': 'all_reduce_perf',
        'algo': ['Auto'],
        'proto': ['Auto'],
        'ranks': [2, 4],
        'data_type': ['float'],
        'data_size': [str(512 * 1024 + 16), str(6882560 * 4)],
        'iters': [2],
        'threads': [2],
        'nprocs': [1],
        'nnodes': [1]}]
}


PPUCaseDict = {
    # ppu case
    'PPUCases': [
        {
            'op': ['alltoall_perf', 'gather_perf', 'scatter_perf', 'sendrecv_perf', 'hypercube_perf'],
            'algo': ['Ring'],
            'proto': ['SimpleBulk'],
            'ranks': [8],
            'data_type': ['int8'],
            'data_size': ['32MB', '102912KB', '512MB', '1229MB'],
            'iters': [10]
        },
        {
        'op': ['all_reduce_perf'],
        'algo': ['Ring', 'Tree'],
        'proto': ['LL', 'LL128', 'Simple', 'SimpleBulk', 'Signal', 'SignalBulk'],
        'ranks': [8],
        'data_type': ['half', 'float', 'double'],
        'data_size':  ['32MB', '102912KB', '512MB', '1229MB'],

        },
        {
        'op': ['all_gather_perf', 'broadcast_perf'],
        'algo': ['Ring'],
        'proto': ['LL', 'LL128', 'Simple'],
        'ranks': [8],
        'data_type': ['int8'],
        'data_size':  ['32MB', '102912KB', '512MB', '1229MB'],
        'iters': [10]
        },
        {
            'op': ['reduce_perf', 'reduce_scatter_perf'],
            'algo': ['Ring'],
            'proto': ['LL', 'LL128', 'Simple'],
            'ranks': [8],
            'data_type': ['half', 'float', 'double'],
            'data_size':  ['32MB', '102912KB', '512MB', '1229MB'],
            'iters': [10]
        }
    ]
}


# attention, nnodes must have the same size with nranks to demonstrate ranks distribution
PPUHwPerfCaseDict = {
    'HWPerf': [{
        'op': ['all_reduce_perf', 'all_gather_perf', 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf',
               'alltoall_perf', 'gather_perf', 'scatter_perf', 'sendrecv_perf', 'hypercube_perf'],
        'algo': ['Auto'],
        'proto': ['Auto'],
        'ranks':  [2, 4, 8, 16, 16, 32, 32, 64],
        'nnodes': [1, 1, 1, 1,  2,  2,  4,  4],
        'data_size': ['1KB', '4KB', '16KB', '64KB', '256KB',
                      '1MB', '2MB', '4MB', '16MB', '32MB', '128MB',
                      '1GB', '1.3GB'],
        'data_type': ['half'],
        'iters': [20],
        'threads': [1]
    }, ],
}

CtaPolicyCaseSM80Dict = {
    'function': [{
        'op': ['all_reduce_perf', 'broadcast_perf', 'alltoall_perf'],
        'algo': ['Auto'],
        'proto': ['Auto'],
        'ranks':  [16],
        'data_size': ['1KB', '1MB', '4MB', '16MB', '1GB'],
        'data_type': ['half'],
        'iters': [20],
        'threads': [1],
        'cta_policy': [1]
    }, ]
}

CtaPolicyCaseSM89Dict = {
    'function': [{
        'op': ['all_reduce_perf', 'broadcast_perf', 'alltoall_perf'],
        'algo': ['Auto'],
        'proto': ['Auto'],
        'ranks':  [2, 4, 8],
        'data_size': ['1KB', '1MB', '4MB', '16MB', '1GB'],
        'data_type': ['half'],
        'iters': [20],
        'threads': [1],
        'cta_policy': [1]
    }, ]
}

CaseDumpSplitMaskAverage = {
    'AverageDict': {
        'op': ['all_reduce_perf'],
        'algo': ['Ring'],
        'redop': ['sum'],
        'proto': ['SimpleBulk'],
        'data_type': ['bf16'],
        'data_size': ['32MB'],
        'iters': [10],
    },
    'MaskDict': {
        'op': ['all_reduce_perf', 'alltoall_perf'],
        'algo': ['Ring'],
        'redop': ['sum'],
        'proto': ['SimpleBulk', 'Auto'],
        'data_type': ['int8'],
        'data_size': ['32MB'],
        'iters': [10],
    },
}


ConfigDict = {
    'PPUCaseDump': {
        'devices_list': ["0,1", "2,3", "6,7", "0,3", "4,7",
                         "0,4", "0,1,2,3", "4,5,6,7", "0,2,4,6", "0,1,4,5"],
        'nproc_threads_list': [[1, 1], [1, 8], [1, 4], [4, 2], [8, 1]],
    },
    'CaseDump': {
        'mask': ['0x01', '0x02', '0x03', '0x04', '0x07', '0x08', '0x0F'],
        'average': [0, 1, 2, 3],
    }
}
