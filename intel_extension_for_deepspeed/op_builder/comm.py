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
        return [
            sycl_kernel_include('csrc/includes'),
        ]

    def is_compatible(self, verbose=True):
        # TODO: add soft compatibility check for private binary release.
        #  a soft check, as in we know it can be trivially changed.
        return super().is_compatible(verbose)

    def extra_ldflags(self):
        ccl_root_path = os.environ.get("CCL_ROOT")
        if ccl_root_path == None:
            raise ValueError(
                "Didn't find CCL_ROOT, install oneCCL from https://github.com/oneapi-src/oneCCL and source its environment variable"
            )
            return []
        else:
            return ['-lccl', f'-L{ccl_root_path}/lib']
