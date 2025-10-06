#pragma once
#include <torch/extension.h>
#include <vector>
#include <tuple>

//--------------------------------------------------------------
// 定义pytorch类型别名

using Tensor = torch::Tensor;
using tensor_list = torch::autograd::tensor_list;
using AutogradContext = torch::autograd::AutogradContext;

//--------------------------------------------------------------
//声明GAT前向和后向函数

Tensor gat_forward_cuda(const Tensor &x, const Tensor &adj, const Tensor &W, const Tensor &a, float alpha);

// std::tuple<Tensor, Tensor, Tensor> gat_backward_cuda(
//     const Tensor& grad_out, const Tensor& x, const Tensor& adj, const Tensor& W, const Tensor& a, const Tensor& out);

// //--------------------------------------------------------------    
// //自定义autograd::Function,用于将自定义模型融入pytorch的梯度图

// struct GATFunction : public torch::autograd::Function<GATFunction> {

//     static Tensor forward(AutogradContext *ctx,
//                         const Tensor &x,
//                         const Tensor &adj,
//                         const Tensor &W,
//                         const Tensor &a,
//                         const float alpha
//                     );

//     static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
// };

//--------------------------------------------------------------    
//定义pytorch模型接口，参数将在这里初始化并注册

// struct GATLayerImpl : public torch::nn::Module {
//     Tensor W; 
//     Tensor a;
//     float alpha;

//     GATLayerImpl(int in_features, int out_features, float alpha = 0.01);

//     Tensor forward(const Tensor &x, const Tensor &adj);
// };
// TORCH_MODULE(GATLayer); // 自动产生 GATLayer (shared_ptr)

//-------------------------------------------------------------- 