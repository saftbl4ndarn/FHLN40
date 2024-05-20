import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs #use lhs to create validation set

E = 1
v = 0.25

def exact_solution(x1, x2):
    u1 = 0.11*x1 + 0.12*x2 + 0.13*x1*x2
    u2 = 0.21*x1 + 0.22*x2 + 0.23*x1*x2
    return [u1, u2]

def analytical(x1, x2, E, v):
    eps1 = 0.11 + 0.13*x2
    eps2 = 0.22 + 0.23*x1
    gam12 = 0.33 + 0.13*x1 + 0.23*x2

    eps = [eps1.detach().numpy(), eps2.detach().numpy(), gam12.detach().numpy()]

    sig1 = E/(1+v) * (eps1 + v/(1-2*v)*(eps1+eps2))
    sig2 = E/(1+v) * (eps2 + v/(1-2*v)*(eps1+eps2))
    sig3 = E*v/(1+v)/(1-2*v) * (eps1 + eps2)
    tau12 = 0.5*E/(1+v) * gam12

    sig = [sig1.detach().numpy(), sig2.detach().numpy(), tau12.detach().numpy()]

    ds1dx1 = E*v/(1+v)/(1-2*v) * 0.23
    ds2dx2 = E*v/(1+v)/(1-2*v) * 0.13
    dt12dx1 = 0.5*E/(1+v) * 0.13
    dt12dx2 = 0.5*E/(1+v) * 0.23

    div1 = ds1dx1 + dt12dx2
    div2 = ds2dx2 + dt12dx1
    return [div1, div2]

def autodiff(x1, x2, u, E, v):
    u1_vals = u[:, 0]
    u2_vals = u[:, 1]

    # Compute gradients
    eps1_x1 = torch.autograd.grad(u1_vals, x1, torch.ones_like(u1_vals), create_graph=True, retain_graph=True)[0]
    eps1_x2 = torch.autograd.grad(u1_vals, x2, torch.ones_like(u1_vals), create_graph=True, retain_graph=True)[0]

    eps2_x1 = torch.autograd.grad(u2_vals, x1, torch.ones_like(u2_vals), create_graph=True, retain_graph=True)[0]
    eps2_x2 = torch.autograd.grad(u2_vals, x2, torch.ones_like(u2_vals), create_graph=True, retain_graph=True)[0]

    # Strain components
    eps1 = eps1_x1
    eps2 = eps2_x2
    gam12 = eps1_x2 + eps2_x1

    eps = [eps1.detach().numpy(), eps2.detach().numpy(), gam12.detach().numpy()]

    sig1 = E/(1+v) * (eps1 + v/(1-2*v)*(eps1+eps2))
    sig2 = E/(1+v) * (eps2 + v/(1-2*v)*(eps1+eps2))
    sig3 = E*v/(1+v)/(1-2*v) * (eps1 + eps2)
    tau12 = 0.5*E/(1+v) * gam12

    sig = [sig1.detach().numpy(), sig2.detach().numpy(), tau12.detach().numpy()]
    
    sig1_x1 = torch.autograd.grad(sig1, x1, torch.ones_like(sig1), create_graph=True, retain_graph=True)[0]
    sig2_x2 = torch.autograd.grad(sig2, x2, torch.ones_like(sig2), create_graph=True, retain_graph=True)[0]
    tau12_x1 = torch.autograd.grad(tau12, x1, torch.ones_like(tau12), create_graph=True, retain_graph=True)[0]
    tau12_x2 = torch.autograd.grad(tau12, x2, torch.ones_like(tau12), create_graph=True, retain_graph=True)[0]

    div1 = sig1_x1 + tau12_x2
    div2 = sig2_x2 + tau12_x1
    return [div1, div2]

#define the neural network
class Network(torch.nn.Module):

    #declare all layers in the constructor 
    def __init__(self, inputdim, hidden1dim, hidden2dim, outputdim):
        super(Network, self).__init__()
        self.input = torch.nn.Linear(inputdim, hidden1dim)
        self.hidden1 = torch.nn.Linear(hidden1dim, hidden2dim)
        self.hidden2 = torch.nn.Linear(hidden1dim, hidden2dim)
        self.output = torch.nn.Linear(hidden2dim, outputdim)

    #in the forward function define how the model is going to be run
    def forward(self, x):

        x = self.input(x) #linear transformation
        x = torch.tanh(x) #tanh as activation function
        x = self.hidden1(x)
        x = torch.tanh(x) #tanh as activation function
        x = self.hidden2(x)
        x = torch.tanh(x)
        x = self.output(x) #linear transformation
        return x
    
    
    
#define the output functions, i.e. the displacements
def u1_fn(x1,x2):
    u1 = 0.11*x1+0.12*x2+0.13*x1*x2
    return u1

def u2_fn(x1,x2):
    u2 = 0.21*x1+0.22*x2+0.23*x1*x2
    return u2

#set up the input and output data as well as mesh
x1 = torch.linspace(0,2,20, requires_grad=True) #change nbr of points when code is running!!!
x2 = torch.linspace(0,1,20, requires_grad=True)
x1_mesh, x2_mesh = torch.meshgrid(x1,x2,indexing="ij")
x = torch.stack((x1_mesh, x2_mesh), dim=2).view(-1,2)
u1 = u1_fn(x1_mesh, x2_mesh) 
u2 = u2_fn(x1_mesh, x2_mesh)
y = torch.stack((u1, u2), dim=2).view(-1,2)

#set up validation set
val_set = torch.tensor(lhs(2,5), dtype=torch.float32)
x1_val_mesh, x2_val_mesh = torch.meshgrid(val_set[:,0], val_set[:,1], indexing="ij")
xval = torch.stack((x1_val_mesh, x2_val_mesh), dim=2).view(-1,2)
u1_val = u1_fn(x1_val_mesh, x2_val_mesh) 
u2_val = u2_fn(x1_val_mesh, x2_val_mesh)
y_val = torch.stack((u1_val, u2_val), dim=2).view(-1,2)

#create the network
torch.manual_seed(2)
inputdim = 2 #nbr of inputs - x1 & x2
hidden1dim = 20 #nbr neurons in first hidden layer
hidden2dim = 20 #nbr neurons in second hidden layer
outputdim = 2 #nbr of outputs - u1 & u2
model = Network(inputdim, hidden1dim, hidden2dim, outputdim)

#define the optimizer and cost function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
cost_fnc = torch.nn.MSELoss()

cost_diff = None
cost_boundary = None

x_boundary = torch.tensor([[0., 0., 2., 2.], [0., 1., 0., 1.]], requires_grad=True).T
u_boundary_exact = torch.stack(exact_solution(x_boundary[:, 0], x_boundary[:, 1])).T

div_exact = torch.tensor(analytical(x[:, 0], x[:, 1], E, v))

#train the network
n_epochs = 2101
lambda1 = 1
lambda2 = 1
training_cost = []
training_cost_diff = []
training_cost_boundary = []

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # Boundary
    u_boundary = model(x_boundary)
    cost_boundary = cost_fnc(u_boundary, u_boundary_exact)

    # Diff
    u = model(x)
    div = autodiff(x1_mesh, x2_mesh, u, E, v)
    div = torch.stack(div, dim=2).view(-1,2)
    
    cost_diff = cost_fnc(div, div_exact)

    cost = lambda1 * cost_boundary + lambda2 * cost_diff
    training_cost.append(cost.item())
    training_cost_diff.append(cost_diff.item())
    training_cost_boundary.append(cost_boundary.item())
    cost.backward(retain_graph=True)

    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Training Cost {cost.item():10.6e}")

plt.plot(range(n_epochs), training_cost, label='Training Cost')
plt.plot(range(n_epochs), training_cost_diff, label='Diff Training Cost')
plt.plot(range(n_epochs), training_cost_boundary, label='Boundary Training Cost')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""Plot displacements."""
x1_plot = x1_mesh.detach().numpy()
x2_plot = x2_mesh.detach().numpy()
u_pred = model(x).detach().numpy()
u_pred = u_pred.reshape((20, 20, 2)) # Reshape to match the meshgrid shape

# Set up plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

# Plot respective surface graphs
surf1 = ax1.plot_surface(x1_plot, x2_plot, u1.detach().numpy(), cmap='jet')
ax1.set_title("Displacements for manufactured solution u1(x1,x2)")
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('u1')
fig.colorbar(surf1, ax=ax1)

surf2 = ax2.plot_surface(x1_plot, x2_plot, u_pred[:, :, 0], cmap='jet')
ax2.set_title("Displacements for trained solution u1(x1,x2)")
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('u1')
fig.colorbar(surf2, ax=ax2)

surf3 = ax3.plot_surface(x1_plot, x2_plot, u2.detach().numpy(), cmap='jet')
ax3.set_title("Displacements for manufactured solution u2(x1,x2)")
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('u2')
fig.colorbar(surf3, ax=ax3)

surf4 = ax4.plot_surface(x1_plot, x2_plot, u_pred[:, :, 1], cmap='jet')
ax4.set_title("Displacements for trained solution u2(x1,x2)")
ax4.set_xlabel('x1')
ax4.set_ylabel('x2')
ax4.set_zlabel('u2')
fig.colorbar(surf4, ax=ax4)

plt.show()