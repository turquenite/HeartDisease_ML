import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.activation_function = torch.nn.ELU()
        self.last_layer_acivation_function = torch.nn.Sigmoid()

        self.linear_1 = torch.nn.Linear(11, 64)
        self.linear_2 = torch.nn.Linear(64, 255)
        self.linear_3 = torch.nn.Linear(255, 64)
        self.linear_4 = torch.nn.Linear(64, 255)
        self.linear_5 = torch.nn.Linear(255, 128)
        self.linear_6 = torch.nn.Linear(128, 64)
        self.linear_7 = torch.nn.Linear(64, 32)
        self.linear_8 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation_function(x)
        x = self.linear_2(x)
        x = self.activation_function(x)
        x = self.linear_3(x)
        x = self.activation_function(x)
        x = self.linear_4(x)
        x = self.activation_function(x)
        x = self.linear_5(x)
        x = self.activation_function(x)
        x = self.linear_6(x)
        x = self.activation_function(x)
        x = self.linear_7(x)
        x = self.activation_function(x)
        x = self.linear_8(x)
        x = self.last_layer_acivation_function(x)

        x = torch.squeeze(x)

        return x