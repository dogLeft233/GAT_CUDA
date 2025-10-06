from gat_cuda_extension import gat_forward_cuda
import torch

device = 'cuda'
B = 32
N = 32
I = 512
O = 512

W = torch.randn(I, O, device=device, dtype=torch.float)
a = torch.randn(2 * O, device=device, dtype=torch.float)
x = torch.randn(B, N , I, device=device, dtype=torch.float)
adj = torch.ones(B, N, N, device=device, dtype=torch.int)
alpha = 0.2

out = gat_forward_cuda(x, adj, W, a, alpha)
print(out)