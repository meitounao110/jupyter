import os
import torch

# f = open('./4001-8000label.txt', 'w')
# for i in range(4001, 8001):
#     result=f'Example{i}.mat\n'
#     f.write(result)
# f.close()

a=torch.tensor([1,3])
b=torch.tensor(2)
c=a[1]+b
print(c)

