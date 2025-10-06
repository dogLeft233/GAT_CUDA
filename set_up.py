from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch, os

torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

setup(
    name='gat_cuda_extension',
    ext_modules=[
        CUDAExtension(
            name='gat_cuda_extension',
            sources=['src/bindings.cpp', 'src/gat_layer.cu'],
            include_dirs=['include'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math', '-std=c++17']
            },
            extra_link_args=[
                f'-Wl,-rpath,{torch_lib_path}',
                '-Wl,-rpath,$ORIGIN'
            ]
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)