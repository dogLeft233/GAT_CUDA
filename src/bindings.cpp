#include <torch/extension.h>
#include "gat_layer.h"  

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GAT CUDA Extension - 图注意力网络CUDA扩展模块";

    m.def("gat_forward_cuda",
          &gat_forward_cuda,
          "GAT forward (CUDA).",
          py::arg("x"), py::arg("adj"), py::arg("W"), py::arg("a"), py::arg("alpha"));
    
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "GAT CUDA Extension";
}
