import torch
import numpy as np
import matplotlib as plt

def exact_solution(x1, x2):
    u1 = 0.11*x1 + 0.12*x2 + 0.13*x1*x2
    u2 = 0.21*x1 + 0.22*x2 + 0.23*x1*x2
    return [u1, u2]

# Material parameters
E = 1
v = 0.25

def calc(x1, x2):
    eps1 = 0.11 + 0.13*x2
    eps2 = 0.22 + 0.23*x1
    gam12 = 0.33 + 0.13*x1 + 0.23*x2

    sig1 = E/(1+v) * (eps1 + v/(1-2*v)*(eps1+eps2))
    sig2 = E/(1+v) * (eps2 + v/(1-2*v)*(eps1+eps2))
    sig3 = E*v/(1+v)/(1-2*v) * (eps1 + eps2)
    tau12 = 0.5*E/(1+v) * gam12

    ds1dx1 = E*v/(1+v)/(1-2*v) * 0.23
    ds2dx2 = E*v/(1+v)/(1-2*v) * 0.13
    dt12dx1 = 0.5*E/(1+v) * 0.13
    dt12dx2 = 0.5*E/(1+v) * 0.23

    b1 = -ds1dx1 - dt12dx2
    b2 = -ds2dx2 - dt12dx1
    b = [b1, b2]
    return b

class Network(torch.nn.Module):

    def __init__(self, n_input, n_layers, n_hidden, n_output):
        super(Network, self).__init__()
        self.input = torch.nn.Sequential(*[
                            torch.nn.Linear(n_input, n_hidden),
                            torch.nn.Tanh()
                        ])
        self.hidden = torch.nn.Sequential(*[
                            torch.nn.Sequential(*[
                                torch.nn.Linear(n_hidden, n_hidden),
                                torch.nn.Tanh()]) for _ in range(n_layers-1)
                        ])
        self.output = torch.nn.Sequential(*[
                            torch.nn.Linear(n_hidden, n_output),
                            torch.nn.Tanh()
                        ])
    
    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x

# Set-up
x1 = torch.linspace(0, 1, 100).view(-1, 1)
x2 = torch.linspace(0, 2, 100).view(-1, 1)
x = torch.stack((x1, x2))
u = exact_solution(x1, x2)

# Create network
input = 2
layers = 3
hidden = 32
output = 2

model = Network(input, layers, hidden, output)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
cost_fnc = torch.nn.MSELoss()

epochs = 2001
for epoch in range(epochs):
    optimizer.zero_grad()
    u_pred = model(x)
    cost = cost_fnc(u_pred, u)
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():10.6e}")


