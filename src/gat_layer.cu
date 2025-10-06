#include "gat_layer.h"
#include "utils.cuh"
#include <torch/torch.h>
#include <cmath>

//--------------------------------------------------------------


Tensor gat_forward_cuda(const Tensor &x, const Tensor &adj, const Tensor &W, const Tensor &a, float alpha){
    const int B = x.size(0);
    const int N = x.size(1);
    const int F = x.size(2);
    const int H = W.size(1);

    auto h = torch::zeros({B, N, H}, torch::kFloat32).contiguous().cuda();
    mat_mul_cuda(x, W, h);

    auto h_1 = torch::zeros({B, N}, torch::kFloat32).contiguous().cuda();
    auto h_2 = torch::zeros({B, N}, torch::kFloat32).contiguous().cuda();
    mat_mul_cuda(h, a.index({torch::indexing::Slice(0, H)}), h_1);
    mat_mul_cuda(h, a.index({torch::indexing::Slice(H, 2*H)}), h_2);

    auto h_1_expend = torch::zeros({B, N, N}, torch::kFloat32).contiguous().cuda();
    expand_cuda(h_1, h_1_expend, 1);

    auto h_2_expend = torch::zeros({B, N, N}, torch::kFloat32).contiguous().cuda();
    expand_cuda(h_2, h_2_expend, 2);

    auto e = torch::zeros({B, N, N}, torch::kFloat32).contiguous().cuda();
    plus_cuda(h_1_expend, h_2_expend, e);

    h_1_expend = Tensor();
    h_2_expend = Tensor();

    leaky_relu_cuda(e, e, alpha);

    mask_replace(e, adj, e, - INFINITY);
    auto score = torch::zeros({B, N, N}, torch::kFloat32).contiguous().cuda();
    soft_max_cuda(e, score, 2);
    e = Tensor();

    auto out_feature = torch::zeros({B, N, H}, torch::kFloat32).contiguous().cuda();
    mat_mul_cuda(score, h, out_feature);

    return out_feature;
}

//--------------------------------------------------------------
//GATFunction

// Tensor GATFunction::forward(AutogradContext *ctx,
//                         const Tensor &x,
//                         const Tensor &adj,
//                         const Tensor &W,
//                         const Tensor &a,
//                         const float alpha
//                     )
// {
//     Tensor out = gat_forward_cuda(x, adj, W, a);

//     ctx->save_for_backward({x, adj, W, a, out});
//     ctx->set_requires_grad({x, adj, W, a});

//     return out;
// }

// tensor_list GATFunction::backward(AutogradContext *ctx, tensor_list grad_outputs) {
//     //!!! 未实现

//     tensor_list outputs;
//     return outputs;
// }

//--------------------------------------------------------------
//GATLayerImpl

GATLayerImpl::GATLayerImpl(int in_features, int out_features, float alpha): alpha(alpha){
    auto W = torch::empty({in_features, out_features}, torch::kFloat32).cuda();
    torch::nn::init::kaiming_uniform_(W, std::sqrt(5));

    auto a = torch::empty({2 * out_features}, torch::kFloat32).cuda();
    torch::nn::init::xavier_uniform_(a.view({-1, 1}));
}

Tensor GATLayerImpl::forward(const Tensor &x, const Tensor &adj){
    return gat_forward_cuda(x, adj, W, a, alpha);
}

//--------------------------------------------------------------