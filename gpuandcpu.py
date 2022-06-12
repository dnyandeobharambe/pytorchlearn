import torch
# check GPU
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
print(dev)

# GPU is not set so tensor runs on CPU 
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
# below both a and b are same as it is running on CPU and so tensor as well as numpy access same memory space
print(a)
print(b)

# numpy can not run on GPU 
import numpy
if torch.cuda.is_available():
    device = torch.device("cuda")
    a = torch.ones(5,device=device)
    b = a.numpy()
    print(a)
    print(b)
    

# To convert tensor to numpy you need to make sure tensor is now running on CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    a = torch.ones(5,device=device)
    b = a.cpu().numpy()
    print(b)

# Let tensor use GPU and numpy use CPU
import numpy
if torch.cuda.is_available():
    device = torch.device("cuda")
    a = torch.ones(5,device=device)
    print(a)
    b = a.to("cpu")
    b = b.numpy()
    print(b)
    a.add_(1)
    print(a)
    print(b)



