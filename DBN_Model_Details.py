import torch
from torchsummary import summary

model = torch.nn.Sequential(
	torch.nn.Linear(28, 512),
	torch.nn.Sigmoid(),
	torch.nn.Linear(512, 128),
	torch.nn.Sigmoid(),
	torch.nn.Linear(128, 64),
	torch.nn.Sigmoid(),
	torch.nn.Linear(64, 10),
	torch.nn.Softmax(dim=1),
)

batch_size = 64
summary(model, input_size=(batch_size, 1, 28, 28))
