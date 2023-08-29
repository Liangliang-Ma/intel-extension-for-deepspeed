import os
from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class CCLCommBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_CCL_COMM"
    NAME = "deepspeed_ccl_comm"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.comm.{self.NAME}_op'

    def sources(self):
        return [
            sycl_kernel_path('csrc/comm/ccl.dp.cpp'),
        ]

    def libraries_args(self):
        args = super().libraries_args()
        return args

    def include_paths(self):
        IPEX_CSRC='/home/maliangl/nightly/auto/nite_auto_20230731/llm.devkit/frameworks.ai.pytorch.ipex-gpu/csrc'
        IPEX_GPU='/home/maliangl/nightly/auto/nite_auto_20230731/llm.devkit/frameworks.ai.pytorch.ipex-gpu/csrc/gpu'
        ipex_root_path='/home/maliangl/miniconda3/envs/nite_auto_20230731/lib/python3.9/site-packages/intel_extension_for_pytorch'
        return [
            sycl_kernel_include('csrc/includes'),
            f'{IPEX_CSRC}',
            f'{IPEX_GPU}',
            f'{IPEX_GPU}/aten',
            f'-L{ipex_root_path}',
            f'-L{ipex_root_path}/include',
            f'-L{ipex_root_path}/include/xpu'
        ]

    def is_compatible(self, verbose=True):
        # TODO: add soft compatibility check for private binary release.
        #  a soft check, as in we know it can be trivially changed.
        return super().is_compatible(verbose)

    def extra_ldflags(self):
        ccl_root_path = os.environ.get("CCL_ROOT")
        ipex_root_path = os.environ.get("IPEX_ROOT")
        # return []
        # if ccl_root_path == None:
        #     raise ValueError(
        #         "Didn't find CCL_ROOT, install oneCCL from https://github.com/oneapi-src/oneCCL and source its environment variable"
        #     )
        #     return []
        # else:
        #     return ['-lccl', f'-L{ccl_root_path}/lib', '-fsycl', '-lintel-ext-pt-python', f'-L{ipex_root_path}/lib']
        
        return ['-lccl', 
                f'-L{ccl_root_path}/lib', 
                '-fsycl', 
                '-lintel-ext-pt-python', 
                f'-L{ipex_root_path}/lib']
    
    
    # CC=dpcpp CFLAGS=-fPIC CXX=dpcpp CXXFLAGS=-fPIC DS_BUILD_DEVICE=dpcpp DS_BUILD_CCL_COMM=1 python setup.py develop |& tee build.log

# export TORCH_ROOT=$CONDA_PREFIX/lib/python3.9/site-packages/torch
# export IPEX_ROOT=$CONDA_PREFIX/lib/python3.9/site-packages/intel_extension_for_pytorch
# export LD_LIBRARY_PATH=${IPEX_ROOT}/lib:${TORCH_ROOT}/lib:${CCL_ROOT}/lib:${LD_LIBRARY_PATH}
# export LIBRARY_PATH=${IPEX_ROOT}/lib:${TORCH_ROOT}/lib:${CCL_ROOT}/lib:${LIBRARY_PATH}