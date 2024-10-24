import torch
import numpy as np

real_labels = torch.ones((32, 1), dtype=torch.float)
print(real_labels)

fake_labels = torch.zeros((32, 1), dtype=torch.float)
print(fake_labels)

positive_labels = [[0,]]
negative_labels = [[1,]]
y = np.concatenate([positive_labels, negative_labels], 0)

#把y转换成tensor
y = torch.tensor(y, dtype=torch.float)
print(y)