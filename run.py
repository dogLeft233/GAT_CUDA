from types import CodeType
from read_files import load_cora_dataset, load_citeseer_dataset
from gat_cuda_extension import gat_forward_cuda
import torch
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from collections import Counter

class CUDATimer:
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

class GATCudaLayer:
    """Wrapper for your CUDA extension but allows passing W,a from outside for fair comparison."""
    def __init__(self, W: torch.Tensor, a: torch.Tensor, alpha: float = 0.2):
        # Expect W, a already on cuda
        self.W = W
        self.a = a
        self.alpha = alpha

    def forward(self, x, adj):
        return gat_forward_cuda(x, adj.int(), self.W, self.a, self.alpha)

    def forward_with_timing(self, x, adj):
        with CUDATimer() as t:
            out = self.forward(x, adj)
        return out, t.elapsed

class GATTorchLayer:
    """Pure PyTorch implementation of a single-head GAT forward (on GPU or CPU)."""
    def __init__(self, W: torch.Tensor, a: torch.Tensor, alpha: float = 0.2):
        # W: (in_features, out_features)
        # a: (2 * out_features,)
        # Both should be torch tensors (on correct device) for fair comparison.
        self.W = W
        self.a = a
        self.alpha = alpha

    def forward(self, x, adj):
        # x: (maybe 1, N, in_features) or (N, in_features)
        # adj: (N, N) adjacency mask (0/1). We'll assume self-loops are present or add them.
        # Convert adj if sparse
        if isinstance(adj, torch.Tensor) and adj.layout != torch.strided:
            try:
                adj = adj.to_dense()
            except Exception:
                adj = torch.tensor(adj).to(x.device)

        # ensure float and on same device
        adj = adj.to(dtype=torch.float32, device=x.device)
        x = x.to(device=self.W.device, dtype=self.W.dtype)
        if x.dim() == 3:
            x = x.squeeze(0)
        
        # Linear transform
        h = x @ self.W  # (N, out_features)
        N, out = h.shape

        # attention coefficients: split a into left and right parts
        a_left = self.a[:out]   # (out,)
        a_right = self.a[out:]  # (out,)

        # compute attention scores: e_ij = LeakyReLU( a_l^T h_i + a_r^T h_j )
        # first compute vectors of shape (N,) for a_l^T h and a_r^T h
        f1 = (h * a_left).sum(dim=1)  # (N,)
        f2 = (h * a_right).sum(dim=1) # (N,)

        # broadcast sum to form e matrix: e[i,j] = f1[i] + f2[j]
        e = f1.unsqueeze(1) + f2.unsqueeze(0)  # (N, N)
        e = F.leaky_relu(e, negative_slope=self.alpha)

        # mask with adjacency: set e_ij = -inf where no edge (so softmax becomes zero)
        # make sure diagonal/self-loops included
        if adj.sum() == 0:
            # if adj is all zeros, add identity
            adj = adj + torch.eye(N, device=adj.device)

        # ensure self-loops
        adj = adj.clone()
        adj.fill_diagonal_(1.0)

        neg_inf = -9e15
        e_masked = torch.where(adj > 0, e, torch.full_like(e, neg_inf))

        # softmax over neighbors (dim=1 -> for each i over j)
        alpha_ij = F.softmax(e_masked, dim=1)  # (N, N)

        # compute output: for each node i, sum_j alpha_ij * h_j
        out_h = alpha_ij @ h  # (N, out)

        return out_h

    def forward_with_timing(self, x, adj):
        # For GPU timing we used CUDATimer externally; keep this for compatibility.
        with CUDATimer() as t:
            out = self.forward(x, adj)
        return out, t.elapsed

def prepare_dataset_tensors(features, adj):
    # convert various possible formats to cuda tensors
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)
    features = features.to(device='cuda', dtype=torch.float32)

    # adjacency handling: if it's numpy or other, convert
    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj)
    # if sparse -> to_dense
    if adj.layout != torch.strided:
        try:
            adj = adj.to_dense()
        except Exception:
            adj = adj.to(device='cuda', dtype=torch.float32)
    adj = adj.to(device='cuda', dtype=torch.float32)
    # ensure square
    if adj.dim() != 2:
        raise ValueError("adj must be a 2D matrix")
    return features, adj

def run_and_record_times(num_run: int, layer, features, adj):
    times = []
    # warm-up single call handled by skipping first measurement inside
    for i in range(num_run + 1):
        _, elapsed = layer.forward_with_timing(features, adj)
        if i == 0:
            continue
        times.append(elapsed * 1000.0)
    return times

def run_and_record_times_cpu(num_run: int, layer, features, adj):
    times = []
    # Warm-up
    for i in range(num_run + 1):
        t0 = time.perf_counter()
        _ = layer.forward(features, adj)
        t1 = time.perf_counter()
        if i == 0:
            continue
        times.append((t1 - t0) * 1000.0)
    return times

def plot_three_way_times(times_cuda, times_gpu, times_cpu,
                         label_cuda="CUDA Ext", label_gpu="PyTorch GPU", label_cpu="PyTorch CPU",
                         dataset_name="Dataset", save_path="three_way.png"):
    """
    Plot three-way scatter (cuda extension, pytorch gpu, pytorch cpu) with avg lines and save.
    Expects times lists (in ms). If lengths differ, plot up to min length to align runs.
    """
    L = min(len(times_cuda), len(times_gpu), len(times_cpu))
    a = times_cuda[:L]
    b = times_gpu[:L]
    c = times_cpu[:L]
    runs = list(range(1, L + 1))

    avg_a = sum(a) / L if L > 0 else 0.0
    avg_b = sum(b) / L if L > 0 else 0.0
    avg_c = sum(c) / L if L > 0 else 0.0

    plt.figure(figsize=(10,6))
    plt.scatter(runs, a, marker='o', label=f"{label_cuda} (runs={L})")
    plt.scatter(runs, b, marker='x', label=f"{label_gpu} (runs={L})")
    plt.scatter(runs, c, marker='^', label=f"{label_cpu} (runs={L})")

    plt.axhline(y=avg_a, linestyle='--', label=f"{label_cuda} Avg: {avg_a:.3f} ms")
    plt.axhline(y=avg_b, linestyle=':', label=f"{label_gpu} Avg: {avg_b:.3f} ms")
    plt.axhline(y=avg_c, linestyle='-.', label=f"{label_cpu} Avg: {avg_c:.3f} ms")

    plt.xlabel("Run Number")
    plt.ylabel("Execution Time (ms)")
    plt.title(f"Execution Time per Run - {dataset_name} (three-way)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved three-way comparison plot for {dataset_name} at {save_path}")

if __name__ == "__main__":
    num_run = 100
    print("=" * 80)

    # -------------------- CORA --------------------
    print("Loading Cora dataset...")
    features, adj = load_cora_dataset()
    features, adj = prepare_dataset_tensors(features, adj)
    _, N, in_feat = features.shape

    # create shared weights for fair comparison
    out_features = 1024
    torch.manual_seed(42)
    W_shared = torch.randn(in_feat, out_features, device='cuda', dtype=torch.float32)
    a_shared = torch.rand(2 * out_features, device='cuda', dtype=torch.float32)

    # instantiate layers (GPU/CUDA)
    cuda_layer = GATCudaLayer(W_shared, a_shared, alpha=0.2)
    torch_layer = GATTorchLayer(W_shared, a_shared, alpha=0.2)

    print("Starting CUDA extension timing (Cora)...")
    cora_times_cuda = run_and_record_times(num_run, cuda_layer, features, adj)
    print("Starting PyTorch GPU timing (Cora)...")
    cora_times_torch = run_and_record_times(num_run, torch_layer, features, adj)

    print(f"Cora CUDA avg: {sum(cora_times_cuda)/len(cora_times_cuda):.3f} ms")
    print(f"Cora PyTorch (GPU) avg: {sum(cora_times_torch)/len(cora_times_torch):.3f} ms")

    # PyTorch on CPU timing (Cora)
    print("Starting PyTorch CPU timing (Cora)...")
    features_cpu, adj_cpu = load_cora_dataset()
    if not isinstance(features_cpu, torch.Tensor):
        features_cpu = torch.tensor(features_cpu, dtype=torch.float32)
    if not isinstance(adj_cpu, torch.Tensor):
        adj_cpu = torch.tensor(adj_cpu)
    features_cpu = features_cpu.to(device='cpu', dtype=torch.float32)
    adj_cpu = adj_cpu.to(device='cpu', dtype=torch.float32)
    torch.manual_seed(42)
    W_shared_cpu = torch.randn(in_feat, out_features, device='cpu', dtype=torch.float32)
    a_shared_cpu = torch.rand(2 * out_features, device='cpu', dtype=torch.float32)
    torch_layer_cpu = GATTorchLayer(W_shared_cpu, a_shared_cpu, alpha=0.2)
    cora_times_cpu = run_and_record_times_cpu(num_run, torch_layer_cpu, features_cpu, adj_cpu)
    print(f"Cora PyTorch (CPU) avg: {sum(cora_times_cpu)/len(cora_times_cpu):.3f} ms")

    # three-way plot for Cora
    plot_three_way_times(cora_times_cuda, cora_times_torch, cora_times_cpu,
                         label_cuda="CUDA Ext", label_gpu="PyTorch GPU", label_cpu="PyTorch CPU",
                         dataset_name="Cora", save_path="cora_three_way.png")

    print("=" * 80)

    # -------------------- CITESEER --------------------
    print("Loading Citeseer dataset...")
    features, adj = load_citeseer_dataset()
    features, adj = prepare_dataset_tensors(features, adj)
    _, N, in_feat = features.shape

    # recreate shared weights for citeseer dims
    out_features = 1024
    torch.manual_seed(42)
    W_shared = torch.randn(in_feat, out_features, device='cuda', dtype=torch.float32)
    a_shared = torch.rand(2 * out_features, device='cuda', dtype=torch.float32)

    cuda_layer = GATCudaLayer(W_shared, a_shared, alpha=0.2)
    torch_layer = GATTorchLayer(W_shared, a_shared, alpha=0.2)

    print("Starting CUDA extension timing (Citeseer)...")
    citeseer_times_cuda = run_and_record_times(num_run, cuda_layer, features, adj)
    print("Starting PyTorch GPU timing (Citeseer)...")
    citeseer_times_torch = run_and_record_times(num_run, torch_layer, features, adj)

    print(f"Citeseer CUDA avg: {sum(citeseer_times_cuda)/len(citeseer_times_cuda):.3f} ms")
    print(f"Citeseer PyTorch (GPU) avg: {sum(citeseer_times_torch)/len(citeseer_times_torch):.3f} ms")

    # PyTorch on CPU timing (Citeseer)
    print("Starting PyTorch CPU timing (Citeseer)...")
    features_cpu, adj_cpu = load_citeseer_dataset()
    if not isinstance(features_cpu, torch.Tensor):
        features_cpu = torch.tensor(features_cpu, dtype=torch.float32)
    if not isinstance(adj_cpu, torch.Tensor):
        adj_cpu = torch.tensor(adj_cpu)
    features_cpu = features_cpu.to(device='cpu', dtype=torch.float32)
    adj_cpu = adj_cpu.to(device='cpu', dtype=torch.float32)
    torch.manual_seed(42)
    W_shared_cpu = torch.randn(in_feat, out_features, device='cpu', dtype=torch.float32)
    a_shared_cpu = torch.rand(2 * out_features, device='cpu', dtype=torch.float32)
    torch_layer_cpu = GATTorchLayer(W_shared_cpu, a_shared_cpu, alpha=0.2)
    citeseer_times_cpu = run_and_record_times_cpu(num_run, torch_layer_cpu, features_cpu, adj_cpu)
    print(f"Citeseer PyTorch (CPU) avg: {sum(citeseer_times_cpu)/len(citeseer_times_cpu):.3f} ms")

    # three-way plot for Citeseer
    plot_three_way_times(citeseer_times_cuda, citeseer_times_torch, citeseer_times_cpu,
                         label_cuda="CUDA Ext", label_gpu="PyTorch GPU", label_cpu="PyTorch CPU",
                         dataset_name="Citeseer", save_path="citeseer_three_way.png")

    print("=" * 80)
