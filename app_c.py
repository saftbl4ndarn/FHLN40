#Physics Informed Neural Network used in task 1.4
import torch
import numpy as np
import matplotlib.pyplot as plt

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

    b1 = -sig1_x1 - tau12_x2
    b2 = -sig2_x2 - tau12_x1
    return [b1, b2]

class PINN:
    "A class used to define the physics informed model in task 1.4 "
    
    def __init__(self, E, v, uB1, uB2, b1, b2):
        self.E = E
        self.v = v
        self.uB1 = uB1
        self.uB2 = uB2
        self.b1 = b1
        self.b2 = b2
        self.model = self.buildModel(2, [10,10], 2) #inputs, [#neurons per hidden layer], outputs
        self.differential_equation_cost_history = None
        self.boundary_condition_cost_history = None
        self.total_cost_history = None
        self.optimizer = None
        self.loss = None
        self.u_res = None

    def buildModel(self, input_dimension, hidden_dimension, output_dimension ):
        """Build a NN with given dimensions"""
        torch.manual_seed(2)
        modules = []
        modules.append(torch.nn.Linear(input_dimension, hidden_dimension[0]))
        modules.append(torch.nn.Tanh())

        for i in range(len(hidden_dimension) - 1):
            modules.append(torch.nn.Linear(hidden_dimension[i], hidden_dimension[i + 1]))
            modules.append(torch.nn.Tanh())
    
        modules.append(torch.nn.Linear(hidden_dimension[-1], output_dimension))
    
        model = torch.nn.Sequential(*modules)
    
        return model

    def getDisplacements(self,x):
        """get displacements"""
        u = self.model(x)
        u.requires_grad_(True)
        return u

    def getDivergence(self, x1, x2, u_pred):
        u1 = u_pred[:,0]
        u2 = u_pred[:,1]

        # Compute gradients
        eps1_x1 = torch.autograd.grad(u1, x1, torch.ones_like(u1), create_graph=True)[0]
        eps1_x2 = torch.autograd.grad(u1, x2, torch.ones_like(u1), create_graph=True)[0]
        eps2_x1 = torch.autograd.grad(u2, x1, torch.ones_like(u2), create_graph=True)[0]
        eps2_x2 = torch.autograd.grad(u2, x2, torch.ones_like(u2), create_graph=True)[0]

        # Strain components
        eps1 = eps1_x1
        eps2 = eps2_x2
        gam12 = eps1_x2 + eps2_x1

        sig1 = self.E/(1+self.v) * (eps1 + self.v/(1-2*self.v)*(eps1+eps2))
        sig2 = self.E/(1+self.v) * (eps2 + self.v/(1-2*self.v)*(eps1+eps2))
        tau12 = 0.5*self.E/(1+self.v) * gam12
    
        sig1_x1 = torch.autograd.grad(sig1, x1, torch.ones_like(sig1), create_graph=True)[0]
        sig2_x2 = torch.autograd.grad(sig2, x2, torch.ones_like(sig2), create_graph=True)[0]
        tau12_x1 = torch.autograd.grad(tau12, x1, torch.ones_like(tau12), create_graph=True)[0]
        tau12_x2 = torch.autograd.grad(tau12, x2, torch.ones_like(tau12), create_graph=True)[0]

        div1 = sig1_x1 + tau12_x2
        div2 = tau12_x1 + sig2_x2
        return div1, div2
    
    def newDivergence(self, x, u_pred):
        sig1_x1, sig2_x2, tau12_x1, tau12_x2 = autodiff(x[:,0], x[:,1], u_pred)
        div1 = sig1_x1 + tau12_x2
        div2 = tau12_x1 + sig2_x2
        return div1, div2
    
    
    def costFunction(self, x1, x2, u_pred, uB1, uB2, b1, b2):
        """compute the cost function"""
        div1, div2 = self.getDivergence(x1, x2, u_pred)
        b1 = b1*torch.ones_like(div1)
        b2 = b2*torch.ones_like(div2)
        cost_diff = torch.sum((div1 + b1)**2) + torch.sum((div2 + b2)**2)

        cost_bound = (u_pred[0,0]-uB1[0,0])**2 + (u_pred[-1,0]-uB1[1,0])**2 + (u_pred[0,1]-uB2[0,0])**2 + (u_pred[-1,1]-uB2[1,0])**2
        return cost_diff, cost_bound


    def closure(self):
        """closure function for the optimizer"""
        self.optimizer.zero_grad()
        u_pred = self.getDisplacements(self.x)
        differential_equation_cost, boundary_condition_cost = self.costFunction(self.x1_mesh, self.x2_mesh, u_pred, self.uB1, self.uB2, self.b1, self.b2)
        loss = differential_equation_cost + boundary_condition_cost
        loss.backward(retain_graph=True)
        return loss

    def train(self, samples_x1, samples_x2, epochs, **kwargs):
        """train the model"""
        x1 = torch.linspace(0,2,samples_x1, requires_grad=True)
        x2 = torch.linspace(0,1,samples_x2, requires_grad=True)
        x1_mesh, x2_mesh = torch.meshgrid(x1,x2,indexing="ij")
        self.x1_mesh = x1_mesh
        self.x2_mesh = x2_mesh
        x = torch.stack((x1_mesh, x2_mesh), dim=2).view(-1,2)
        self.x = x

        #Set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)

        #Initialize history arrays
        self.differential_equation_cost_history = np.zeros(epochs)
        self.boundary_condition_cost_history = np.zeros(epochs)
        self.total_cost_history = np.zeros(epochs)
        self.u_res = 0

        #Training loop
        for i in range(epochs):
            # Predict displacements
            u_pred = self.getDisplacements(x)

            # Cost function calculation
            differential_equation_cost, boundary_condition_cost = self.costFunction(x1_mesh, x2_mesh, u_pred, self.uB1, self.uB2, self.b1, self.b2)

            # Total cost
            total_cost = differential_equation_cost + boundary_condition_cost

            # Add energy values to history
            self.differential_equation_cost_history[i] += differential_equation_cost
            self.boundary_condition_cost_history[i] += boundary_condition_cost
            self.total_cost_history[i] += total_cost
            self.u_res = u_pred 

            # Print training state
            self.printTrainingState(i, epochs)

            # Update parameters
            self.optimizer.step(self.closure)
        
        self.x = None


    def printTrainingState(self, epoch, epochs, print_every=100):
        """Print the cost values of the current epoch in a training loop."""

        if epoch == 0 or epoch == (epochs - 1) or epoch % print_every == 0 or print_every == 'all':
            # Prepare string
            string = "Epoch: {}/{}\t\tDifferential equation cost = {:2f}\t\tBoundary condition cost = {:2f}\t\tTotal cost = {:2f}"

            # Format string and print
            print(string.format(epoch, epochs - 1, self.differential_equation_cost_history[epoch],
                                self.boundary_condition_cost_history[epoch], self.total_cost_history[epoch]))

    def plotTrainingHistory(self, yscale='log'):
        """Plot the training history."""

        # Set up plot
        fig, ax = plt.subplots()
        ax.set_title("Cost function history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost function $C$")
        plt.yscale(yscale)

        # Plot data
        ax.plot(self.differential_equation_cost_history, label="Differential equation cost")
        ax.plot(self.boundary_condition_cost_history, label="Boundary condition cost")
        ax.plot(self.total_cost_history, label="Total cost")

        ax.legend()

    def plotDisplacements(self, u1_analytic, u2_analytic):
        """Plot displacements."""
        x1_plot = self.x1_mesh.detach().numpy()
        x2_plot = self.x2_mesh.detach().numpy()

        # Set up plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

        #plot respective surface graphs
        surf1 = ax1.plot_surface(x1_plot, x2_plot, u1_analytic, cmap='jet')
        ax1.set_title("Displacements for manufactured solution u1(x1,x2)")
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('u1')
        fig.colorbar(surf1, ax=ax1)
        
        surf2 = ax2.plot_surface(x1_plot, x2_plot, self.u_res[:,0].view(-1,2).reshape(x1_plot.shape).detach().numpy(), cmap='jet')
        ax2.set_title("Displacements for trained solution u1(x1,x2)")
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('u1')
        fig.colorbar(surf2, ax=ax2)
        
        surf3 = ax3.plot_surface(x1_plot, x2_plot, u2_analytic, cmap='jet')
        ax3.set_title("Displacements for manufactured solution u2(x1,x2)")
        ax3.set_xlabel('x1')
        ax3.set_ylabel('x2')
        ax3.set_zlabel('u2')
        fig.colorbar(surf3, ax=ax3)
        
        surf4 = ax4.plot_surface(x1_plot, x2_plot, self.u_res[:,1].view(-1,2).reshape(x1_plot.shape).detach().numpy(), cmap='jet')
        ax4.set_title("Displacements for trained solution u2(x1,x2)")
        ax4.set_xlabel('x1')
        ax4.set_ylabel('x2')
        ax4.set_zlabel('u2')
        fig.colorbar(surf4, ax=ax4)