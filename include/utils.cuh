#pragma once

#include<cmath>

//--------------------------------------------------------------
// GAT CUDA工具函数库
// 主要是一些数学运算的cuda优化实现
//--------------------------------------------------------------

/*
函数列表：

核函数 (__global__):
- mat_mul_cuda_kernel: 矩阵乘法核函数，支持批处理，使用共享内存优化
- mat_vec_cuda_kernel: 矩阵向量乘法核函数
- plus_cuda_kernel: 向量加法核函数
- mul_cuda_kernel: 向量乘法核函数
- exp_cuda_kernel: 指数函数核函数
- cat_cuda_kernel: 张量拼接核函数
- expand_cuda_kernel: 张量扩展核函数
- leaky_relu_cuda_kernel: LeakyReLU激活函数核函数
- softmax_cuda_kernel: Softmax激活函数核函数
- sum_cuda_kernel: 张量求和核函数
- max_cuda_kernel: 张量最大值核函数
- calculate_attn_cuda: 注意力计算核函数
- mask_replace_kernel: 掩码替换核函数
- num_mul_cuda_kernel: 标量乘法核函数

主机接口函数 (__host__):
- mat_mul_cuda: 矩阵乘法主机接口
- plus_cuda: 向量加法主机接口
- mul_cuda: 向量乘法主机接口
- exp_cuda: 指数函数主机接口
- cat_cuda: 张量拼接主机接口
- expand_cuda: 张量扩展主机接口
- leaky_relu_cuda: LeakyReLU激活函数主机接口
- soft_max_cuda: Softmax激活函数主机接口
- sum_cuda: 张量求和主机接口
- max_cuda: 张量最大值主机接口
- mask_replace: 掩码替换主机接口
- num_mul_cuda: 标量乘法主机接口

工具函数:
- check_cuda_contiguous: 检查张量是否在CUDA上且连续
- get_N: 获取张量元素总数
*/

//--------------------------------------------------------------
// 用于二维线程块的长和宽
#define TILE 16

// 用于一维线程块长度
#define LINE_TILE 64

using Tensor = torch::Tensor;

//--------------------------------------------------------------
//核函数区，其中的函数不推荐未经包装在外部调用。


//张量乘法，支持batch，使用共享内存优化。
//线程块为二维，网格为三维，用于处理batch
__global__ void mat_mul_cuda_kernel(
    const float* A,
    const float* B,
    float* C,
    int batch,
    int M,
    int N,
    int K,
    int b_batch
) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int b = blockIdx.z;

    if (b >= batch) return;

    const float* A_batch = A + b * M * K;
    const float* B_batch = B + ( (b_batch==1?0:b) ) * K * N;
    float* C_batch = C + b * M * N;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        if (row < M && t*TILE + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A_batch[row*K + t*TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (t*TILE + threadIdx.y < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B_batch[(t*TILE + threadIdx.y)*N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C_batch[row*N + col] = sum;
}

//张量和向量乘法
__global__ void mat_vec_cuda_kernel(
    const float* A,
    const float* B,
    float* C,
    int batch,
    int M,
    int K,
    int b_batch
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (row >= M || b >= batch) return;

    const float* A_batch = A + b * M * K;
    const float* B_batch = B + ( (b_batch==1?0:b) ) * K;
    float* C_batch = C + b * M;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A_batch[row*K + k] * B_batch[k];
    }
    C_batch[row] = sum;
}

__global__ void plus_cuda_kernel(const float* A, const float* B, float* C, int N){
    int bid =  blockIdx.x;
    int tid = threadIdx.x;
    int id = bid * blockDim.x + tid;

    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

__global__ void mul_cuda_kernel(const float* A, const float* B, float* C, int N){
    int bid =  blockIdx.x;
    int tid = threadIdx.x;
    int id = bid * blockDim.x + tid;

    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < N; i += stride) {
        C[i] = A[i] * B[i];
    }
}

__global__ void exp_cuda_kernel(const float* A, float* B, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= N)return;

    B[id]  = exp(A[id]);
}

__global__ void cat_cuda_kernel(const float* A, const float* B, float* C,
                    int total, int a_stride, int b_stride) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= total) return;

    int c_block = a_stride + b_stride;
    int c_bid = id / c_block;
    int c_tid = id % c_block;

    if (c_tid < a_stride) {
        C[id] = A[c_bid * a_stride + c_tid];
    } else {
        C[id] = B[c_bid * b_stride + (c_tid - a_stride)];
    }
}

__global__ void expand_cuda_kernel(const float* A, float* B, int N, int time_, int stride){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= N)return;

    int b_block = id / (time_ * stride);
    int b_id = id % (time_ * stride);
    int a_id = b_block * stride + b_id % stride;

    B[id] = A[a_id];
}

__global__ void leaky_relu_cuda_kernel(const float* A, float* B, float alpha, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < N; i += stride) {
        float x = A[i];
        B[i] = (x >= 0.0f) ? x : alpha * x;
    }
}

__global__ void softmax_cuda_kernel(const float* A_exp,const float* A_exp_sum, float* B, int total, int stride){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= total) return;

    int a_block = id / stride;

    B[id] = A_exp[id] / A_exp_sum[a_block];
}

__global__ void sum_cuda_kernel(const float* A, float* B, int outer, int dim_len, int stride){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= outer * stride) return;

    int outer_idx = id / stride;
    int inner_idx = id % stride;

    float sum = 0;
    for (int k = 0; k < dim_len; k++) {
        int offset = outer_idx * dim_len * stride + k * stride + inner_idx;
        sum += A[offset];
    }

    B[id] = sum;
}

__global__ void max_cuda_kernel(const float* A, float* B,
                                int outer, int dim_len, int stride) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= outer * stride) return;

    int outer_idx = id / stride;
    int inner_idx = id % stride;

    float max_val = -INFINITY;
    for (int k = 0; k < dim_len; k++) {
        int offset = outer_idx * dim_len * stride + k * stride + inner_idx;
        max_val = fmaxf(max_val, A[offset]);
    }

    B[id] = max_val;
}


__global__ void calculate_attn_cuda(const float* h, const float* a, float* e, int B, int N, int H){
    int id =blockIdx.z * blockDim.x * blockDim.z + blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= B * N * N)return;

    int batch = id / (N * N);
    int y = id % (N * N) / N;
    int x = id % (N * N) % N;

    float sum =0.0;
    for(int i=0 ;i < H; ++i)
        sum += h[batch * N * H + x * H + i] * a[i] + h[batch * N * H + y * H + i] * a[i + H];
    e[id] = sum;
}

__global__ void mask_replace_kernel(const float* A, const int* adj, float* B,const float zero_val, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= N)return;

    B[id] = (adj[id] == 0) ? zero_val : A[id];
}

__global__ void num_mul_cuda_kernel(const float* A, const float num, float* B, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= N)return;

    B[id] = num * A[id];
}

//--------------------------------------------------------------
//对外接口区，推荐在主机中调用这些函数。

inline void check_cuda_contiguous(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.device().is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

__host__ int get_N(Tensor a){
    int N =1;
    for(int i=0;i<a.dim();++i)
        N*=a.size(i);
    return N;
}

__host__ void mat_mul_cuda(Tensor a, Tensor b, Tensor c) {
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");
    check_cuda_contiguous(c, "Tensor c");

    int M = (a.dim() >= 2) ? a.size(-2) : 1;
    int K = a.size(-1);
    int N = (b.dim() >= 2) ? b.size(-1) : 1;

    int a_batch = 1;
    for (int i = 0; i < a.dim() - 2; ++i) a_batch *= a.size(i);

    int b_batch = 1;
    for (int i = 0; i < b.dim() - 2; ++i) b_batch *= b.size(i);

    if (b.dim() == 1) {
        // 张量 × 向量
        dim3 block(LINE_TILE);
        dim3 grid((M + LINE_TILE - 1) / LINE_TILE, a_batch);
        mat_vec_cuda_kernel<<<grid, block>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            a_batch, M, K, b_batch
        );
    } else {
        // 张量 × 张量
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1)/TILE, (M + TILE - 1)/TILE, a_batch);
        mat_mul_cuda_kernel<<<grid, block>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            a_batch, M, N, K, b_batch
        );
    }

    cudaDeviceSynchronize();
}

__host__ void plus_cuda(Tensor a, Tensor b, Tensor c){
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");
    check_cuda_contiguous(c, "Tensor c");

    int N = get_N(a);

    dim3 block(LINE_TILE);
    dim3 grid((N + LINE_TILE -1 )/ LINE_TILE);

    plus_cuda_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();
}

__host__ void exp_cuda(Tensor a, Tensor b){
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");

    int N = get_N(a);

    dim3 block(LINE_TILE);
    dim3 grid((N + LINE_TILE -1 )/ LINE_TILE);

    exp_cuda_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();
}

__host__ void mul_cuda(Tensor a, Tensor b, Tensor c){
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");
    check_cuda_contiguous(c, "Tensor c");

    int N = get_N(a);

    dim3 block(LINE_TILE);
    dim3 grid((N + LINE_TILE -1 )/ LINE_TILE);

    mul_cuda_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        N
    );
    
    cudaDeviceSynchronize();
}

__host__ void cat_cuda(Tensor a, Tensor b, Tensor c, int dim){
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");
    check_cuda_contiguous(c, "Tensor c");

    int a_stride = 1;
    for(int i=a.dim()-1;i>=dim;--i)
        a_stride *= a.size(i);

    int b_stride = 1;
    for(int i=b.dim()-1;i>=dim;--i)
        b_stride *= b.size(i);

    int a_N = get_N(a);
    int b_N = get_N(b);
    int N = a_N + b_N;

    dim3 block(LINE_TILE);
    dim3 grid((N + LINE_TILE -1 )/ LINE_TILE);

    cat_cuda_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        N, a_stride, b_stride
    );

    cudaDeviceSynchronize();
}

__host__ void leaky_relu_cuda(Tensor a, Tensor b, float alpha){
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");

    int N = get_N(a);

    dim3 block(LINE_TILE);
    dim3 grid((N + LINE_TILE -1 )/ LINE_TILE);
    leaky_relu_cuda_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        alpha, N
    );

    cudaDeviceSynchronize();
}

__host__ void expand_cuda(Tensor a, Tensor b, int dim){
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");

    int stride = 1;
    for(int i = a.dim() - 1; i>=dim; --i)
        stride *= a.size(i);

    int time_ = b.size(dim);
    int N = get_N(b);

    dim3 block(LINE_TILE);
    dim3 grid((N + LINE_TILE -1 )/ LINE_TILE);
    expand_cuda_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        N, time_, stride
    );

    cudaDeviceSynchronize();
}

__host__ void sum_cuda(Tensor a, Tensor b, int dim){
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");

    int outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= a.size(i);
    }

    int stride = 1;
    for (int i = dim + 1; i < a.dim(); i++) {
        stride *= a.size(i);
    }

    int total = outer * stride;

    dim3 block(LINE_TILE);
    dim3 grid((total + LINE_TILE - 1) / LINE_TILE);

    sum_cuda_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        outer, a.size(dim), stride
    );

    cudaDeviceSynchronize();
}

__host__ void max_cuda(Tensor a, Tensor b, int dim) {
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");

    int outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= a.size(i);
    }

    int stride = 1;
    for (int i = dim + 1; i < a.dim(); i++) {
        stride *= a.size(i);
    }

    int total = outer * stride;

    dim3 block(LINE_TILE);
    dim3 grid((total + LINE_TILE - 1) / LINE_TILE);

    max_cuda_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        outer, a.size(dim), stride
    );

    cudaDeviceSynchronize();
}

__host__ void mask_replace(Tensor a, Tensor adj, Tensor b, float zero_val){
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");
    check_cuda_contiguous(adj, "Tensor adj");

    int N = get_N(a);

    dim3 block(LINE_TILE);
    dim3 grid((N + LINE_TILE -1 )/ LINE_TILE);
    mask_replace_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        adj.data_ptr<int>(),
        b.data_ptr<float>(),
        zero_val, N
    );

    cudaDeviceSynchronize();
}

__host__ void num_mul_cuda(Tensor a, Tensor b, float num){
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");

    int N = get_N(a);

    dim3 block(LINE_TILE);
    dim3 grid((N + LINE_TILE -1 )/ LINE_TILE);
    num_mul_cuda_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        num,
        b.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();
}

__host__ void soft_max_cuda(Tensor a, Tensor b, int dim){
    check_cuda_contiguous(a, "Tensor a");
    check_cuda_contiguous(b, "Tensor b");

    int N = get_N(a);
    auto sizes = a.sizes();
    std::vector<int64_t> new_sizes;        
    new_sizes.reserve(sizes.size() - 1); 

    for (int i = 0; i < sizes.size(); i++) {
        if (i != dim) new_sizes.push_back(sizes[i]);
    }

    Tensor a_max = torch::zeros(new_sizes, torch::kFloat32).contiguous().cuda();
    Tensor a_max_extend = torch::zeros(a.sizes(), torch::kFloat32).contiguous().cuda();

    max_cuda(a, a_max, dim);
    num_mul_cuda(a_max, a_max, -1);
    expand_cuda(a_max, a_max_extend, dim);
    a_max = torch::zeros(a.sizes(), torch::kFloat32).contiguous().cuda();
    plus_cuda(a, a_max_extend, a_max);

    Tensor a_exp = torch::zeros(a.sizes(), torch::kFloat32).contiguous().cuda();
    exp_cuda(a_max, a_exp);
    a_max = Tensor();

    Tensor a_exp_sum = torch::zeros(new_sizes, torch::kFloat32).contiguous().cuda();
    sum_cuda(a_exp, a_exp_sum, dim);

    int stride = 1;
    for(int i = a.dim() - 1; i >= dim;--i)
        stride *= a.size(i);

    dim3 block(LINE_TILE);
    dim3 grid((N + LINE_TILE -1 )/ LINE_TILE);
    softmax_cuda_kernel<<<grid, block>>>(
        a_exp.data_ptr<float>(),
        a_exp_sum.data_ptr<float>(),
        b.data_ptr<float>(),
        N, stride
    );

    cudaDeviceSynchronize();
}

//--------------------------------------------------------------