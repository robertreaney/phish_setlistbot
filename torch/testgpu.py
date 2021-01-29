import torch

torch.cuda.is_available() #nice

#create device object
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#move things to GPU with .to(dev)
