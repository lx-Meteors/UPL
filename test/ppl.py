import torch

loss = torch.tensor(2.37984)

ppl = torch.exp(loss)

print(ppl)