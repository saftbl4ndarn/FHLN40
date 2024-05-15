import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.layers = n_layers
        self.input = torch.nn.Linear(n_input, n_hidden)
        self.hidden = torch.nn.Linear(n_hidden, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = self.input(x)
        x = torch.tanh(x)
        for _ in range(self.layers):
            x = self.hidden(x)
            x = torch.tanh(x)
        x = self.output(x)
        return x

# Set-up training data
x1 = torch.linspace(0, 1, 100).view(-1, 1)
x2 = torch.linspace(0, 2, 100).view(-1, 1)
x = torch.meshgrid((x1, x2))
u = torch.stack(exact_solution(x1, x2))

# Set-up validation data

x1_val = torch.linspace(0, 1, 25).view(-1, 1)
x2_val = torch.linspace(0, 1, 25).view(-1, 1)
x_val = torch.cat((x1_val, x2_val), dim=1)
u_val = torch.stack(exact_solution(x1_val, x2_val))

# Create network
input = 2
layers = 2
hidden = 16
output = 2

model = Network(input, layers, hidden, output)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-2, lr=0.001)
cost_fnc = torch.nn.MSELoss()

epochs = 2001
training_cost = []
validation_cost = []
for epoch in range(epochs):
    optimizer.zero_grad()
    u_pred = model(x)
    u = u.view(u_pred.shape)
    cost = cost_fnc(u_pred, u)
    u_pred_val = model(x_val)
    u_val = u_val.view(u_pred_val.shape)
    val_cost = cost_fnc(u_pred_val, u_val)
    
    training_cost.append(cost.item())
    validation_cost.append(val_cost.item())

    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Training Loss {cost.item():10.6e}, Validation Loss {val_cost.item():10.6e}")


plt.plot(range(epochs), training_cost, label='Training Cost')
plt.plot(range(epochs), validation_cost, label='Validation Cost')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Cost over Epochs')
plt.legend()

with torch.no_grad():
    x1_plot, x2_plot = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 2, 100))
    x_plot = torch.cat((x1_plot.reshape(-1, 1), x2_plot.reshape(-1, 1)), dim=1)
    u_pred_plot = model(x_plot).numpy()
    u_exact = torch.meshgrid(torch.stack(exact_solution(x1_plot, x2_plot)))
    u_error = u - u_pred_plot

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x1_plot, x2_plot, u_error[:, 0], c=u_error[:, 0], cmap='RdYlGn')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('u1 prediction error')
    ax1.set_title('Prediction error for u1')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x1_plot, x2_plot, u_error[:, 1], c=u_error[:, 1], cmap='RdYlGn')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('u2 prediction error')
    ax2.set_title('Prediction error for u2')

    plt.show()
