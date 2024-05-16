import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs  # use lhs to create validation set

E = 1
v = 0.25

def exact_solution(x1, x2):
    u1 = 0.11 * x1 + 0.12 * x2 + 0.13 * x1 * x2
    u2 = 0.21 * x1 + 0.22 * x2 + 0.23 * x1 * x2
    return torch.stack((u1, u2), dim=1)

def analytical(x1, x2, E, v):
    eps1 = 0.11 + 0.13 * x2
    eps2 = 0.22 + 0.23 * x1
    gam12 = 0.33 + 0.13 * x1 + 0.23 * x2

    sig1 = E / (1 + v) * (eps1 + v / (1 - 2 * v) * (eps1 + eps2))
    sig2 = E / (1 + v) * (eps2 + v / (1 - 2 * v) * (eps1 + eps2))
    tau12 = 0.5 * E / (1 + v) * gam12

    ds1dx1 = E * v / (1 + v) / (1 - 2 * v) * 0.23
    ds2dx2 = E * v / (1 + v) / (1 - 2 * v) * 0.13
    dt12dx1 = 0.5 * E / (1 + v) * 0.13
    dt12dx2 = 0.5 * E / (1 + v) * 0.23

    b1 = -ds1dx1 - dt12dx2
    b2 = -ds2dx2 - dt12dx1
    return torch.tensor([b1, b2], dtype=torch.float32)

def autodiff(x, u, E, v):
    x1 = x[:, 0].requires_grad_(True)
    x2 = x[:, 1].requires_grad_(True)
    u1_vals = u[:, 0]
    u2_vals = u[:, 1]

    eps1_x1 = torch.autograd.grad(u1_vals, x1, torch.ones_like(u1_vals), create_graph=True, retain_graph=True, allow_unused=True)[0]
    eps1_x2 = torch.autograd.grad(u1_vals, x2, torch.ones_like(u1_vals), create_graph=True, retain_graph=True, allow_unused=True)[0]

    eps2_x1 = torch.autograd.grad(u2_vals, x1, torch.ones_like(u2_vals), create_graph=True, retain_graph=True, allow_unused=True)[0]
    eps2_x2 = torch.autograd.grad(u2_vals, x2, torch.ones_like(u2_vals), create_graph=True, retain_graph=True, allow_unused=True)[0]

    eps1 = eps1_x1 if eps1_x1 is not None else torch.zeros_like(x1)
    eps2 = eps2_x2 if eps2_x2 is not None else torch.zeros_like(x2)
    gam12 = (eps1_x2 if eps1_x2 is not None else torch.zeros_like(x2)) + (eps2_x1 if eps2_x1 is not None else torch.zeros_like(x1))

    sig1 = E / (1 + v) * (eps1 + v / (1 - 2 * v) * (eps1 + eps2))
    sig2 = E / (1 + v) * (eps2 + v / (1 - 2 * v) * (eps1 + eps2))
    tau12 = 0.5 * E / (1 + v) * gam12

    sig1_x1 = torch.autograd.grad(sig1, x1, torch.ones_like(sig1), create_graph=True, retain_graph=True, allow_unused=True)[0]
    sig2_x2 = torch.autograd.grad(sig2, x2, torch.ones_like(sig2), create_graph=True, retain_graph=True, allow_unused=True)[0]
    tau12_x1 = torch.autograd.grad(tau12, x1, torch.ones_like(tau12), create_graph=True, retain_graph=True, allow_unused=True)[0]
    tau12_x2 = torch.autograd.grad(tau12, x2, torch.ones_like(tau12), create_graph=True, retain_graph=True, allow_unused=True)[0]

    b1 = -sig1_x1 if sig1_x1 is not None else torch.zeros_like(x1) - (tau12_x2 if tau12_x2 is not None else torch.zeros_like(x2))
    b2 = -sig2_x2 if sig2_x2 is not None else torch.zeros_like(x2) - (tau12_x1 if tau12_x1 is not None else torch.zeros_like(x1))
    return torch.stack((b1, b2), dim=1)

class Network(torch.nn.Module):
    def __init__(self, inputdim, hidden1dim, hidden2dim, outputdim):
        super(Network, self).__init__()
        self.input = torch.nn.Linear(inputdim, hidden1dim)
        self.hidden1 = torch.nn.Linear(hidden1dim, hidden2dim)
        self.hidden2 = torch.nn.Linear(hidden2dim, hidden2dim)
        self.output = torch.nn.Linear(hidden2dim, outputdim)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = self.output(x)
        return x

def u1_fn(x1, x2):
    return 0.11 * x1 + 0.12 * x2 + 0.13 * x1 * x2

def u2_fn(x1, x2):
    return 0.21 * x1 + 0.22 * x2 + 0.23 * x1 * x2

x1 = torch.linspace(0, 2, 20, requires_grad=True)
x2 = torch.linspace(0, 1, 20, requires_grad=True)
x1_mesh, x2_mesh = torch.meshgrid(x1, x2, indexing="ij")
x = torch.stack((x1_mesh, x2_mesh), dim=2).view(-1, 2)
u1 = u1_fn(x1_mesh, x2_mesh)
u2 = u2_fn(x1_mesh, x2_mesh)
y = torch.stack((u1, u2), dim=2).view(-1, 2)

val_set = torch.tensor(lhs(2, 5), dtype=torch.float32)
x1_val_mesh, x2_val_mesh = torch.meshgrid(val_set[:, 0], val_set[:, 1], indexing="ij")
xval = torch.stack((x1_val_mesh, x2_val_mesh), dim=2).view(-1, 2)
u1_val = u1_fn(x1_val_mesh, x2_val_mesh)
u2_val = u2_fn(x1_val_mesh, x2_val_mesh)
y_val = torch.stack((u1_val, u2_val), dim=2).view(-1, 2)

torch.manual_seed(2)
inputdim = 2  # number of inputs - x1 & x2
hidden1dim = 20  # number of neurons in first hidden layer
hidden2dim = 20  # number of neurons in second hidden layer
outputdim = 2  # number of outputs - u1 & u2
model = Network(inputdim, hidden1dim, hidden2dim, outputdim)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
cost_fnc = torch.nn.MSELoss()

x_boundary = torch.tensor([[0., 0., 2., 2.], [0., 1., 0., 1.]], requires_grad=True).T
u_boundary_exact = exact_solution(x_boundary[:, 0], x_boundary[:, 1])

b_exact = analytical(x[:, 0], x[:, 1], E, v)

n_epochs = 2000
lambda1 = 1e-2
lambda2 = 1e-2

valcost_history = np.zeros(n_epochs)
traincost_history = np.zeros(n_epochs)

for epoch in range(n_epochs):
    #optimizer.zero_grad()

    # Boundary
    u_boundary = model(x_boundary)
    cost_boundary = cost_fnc(u_boundary, u_boundary_exact)

    # Diff
    u = model(x)
    torch.enable_grad()
    b = autodiff(x, u, E, v)
    cost_diff = cost_fnc(b, b_exact)

    cost = lambda1 * cost_boundary + lambda2 * cost_diff

    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Training Cost {cost.item():10.6e}")
