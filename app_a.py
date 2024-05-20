import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs #use lhs to create validation set

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
        x = torch.tanh(self.hidden1(x)) #tanh as activation function
        x = torch.tanh(self.hidden2(x)) #tanh as activation function
        x = self.output(x) #linear transformation

        return x
    
#define the output functions, i.e. the displacements
def u1_fn(x1,x2):
    u1 = 0.11*x1+0.12*x2+0.13*x1*x2
    return u1

def u2_fn(x1,x2):
    u2 = 0.21*x1+0.22*x2+0.23*x1*x2
    return u2

#set up the input and output data aswell as mesh
x1 = torch.linspace(0.,2.,20) 
x2 = torch.linspace(0.,1.,20)
x1_mesh, x2_mesh = torch.meshgrid(x1,x2,indexing="ij")
x = torch.stack((x1_mesh, x2_mesh), dim=2).view(-1,2)
u1 = u1_fn(x1_mesh, x2_mesh) 
u2 = u2_fn(x1_mesh, x2_mesh)
y = torch.stack((u1, u2), dim=2).view(-1,2)

#set up validation set
val_set = torch.tensor(lhs(2, 25), dtype=torch.float32)
val_set[:, 0] = val_set[:, 0] * 2 # Map 0-1 to 0-2
u1_val = u1_fn(val_set[:, 0], val_set[:, 1]) 
u2_val = u2_fn(val_set[:, 0], val_set[:, 1])
y_val = torch.stack((u1_val.T, u2_val.T)).view(-1, 2)
print(val_set.detach().numpy())

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


#train the network
n_epochs = 5000
#initialize history arrays
valcost_history = np.zeros(n_epochs)
traincost_history = np.zeros(n_epochs)
for epoch in range(n_epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    val_pred = model(val_set)
    cost = cost_fnc(y_pred, y)
    val_cost = cost_fnc(val_pred, y_val) #calculate the cost for the validation data 
    cost.backward()
    optimizer.step()
    if epoch%100==0:
        print(f"Epoch {epoch}, Training Cost {cost.item():10.6e} Validation Cost {val_cost.item():10.6e}")

    #insert values in history arrays
    valcost_history[epoch] += cost
    traincost_history[epoch] += val_cost

plt.plot(range(n_epochs), traincost_history)
plt.plot(range(n_epochs), valcost_history)
plt.yscale("log")
plt.show()