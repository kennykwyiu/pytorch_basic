import torch

dev = torch.device("cpu")
dev = torch.device("mps")

a = torch.tensor([2, 2],
                 dtype=torch.float32,
                 device=dev)
print(a)

indices = torch.tensor([[0, 1, 2],  # 0,0 (row ?, col ?), 1,1, 2,2
                        [0, 1, 2]])
values = torch.tensor([1, 2, 3])
x = torch.sparse_coo_tensor(indices, values, [4, 4],
                            dtype=torch.float32,
                            device=dev).to_dense()

print(x)
