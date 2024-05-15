import torch

# Material properties
E = 1
v = 0.25

def analytical(x1, x2):
    eps1 = 0.11 + 0.13*x2
    eps2 = 0.22 + 0.23*x1
    gam12 = 0.33 + 0.13*x1 + 0.23*x2

    eps = [eps1.detach().numpy(), eps2.detach().numpy(), gam12.detach().numpy()]
    print("Analytical strains:")
    for y in eps:
        print(y)

    sig1 = E/(1+v) * (eps1 + v/(1-2*v)*(eps1+eps2))
    sig2 = E/(1+v) * (eps2 + v/(1-2*v)*(eps1+eps2))
    sig3 = E*v/(1+v)/(1-2*v) * (eps1 + eps2)
    tau12 = 0.5*E/(1+v) * gam12

    sig = [sig1.detach().numpy(), sig2.detach().numpy(), tau12.detach().numpy()]
    print("Analytical stresses:")
    for y in sig:
        print(y)

    ds1dx1 = E*v/(1+v)/(1-2*v) * 0.23
    ds2dx2 = E*v/(1+v)/(1-2*v) * 0.13
    dt12dx1 = 0.5*E/(1+v) * 0.13
    dt12dx2 = 0.5*E/(1+v) * 0.23

    div = [ds1dx1, ds2dx2, dt12dx1, dt12dx2]
    print("Analytic div:")
    for y in div:
        print(y)

    b1 = -ds1dx1 - dt12dx2
    b2 = -ds2dx2 - dt12dx1
    b = [b1, b2]
    return b

# Displacement functions
def autograd(x1, x2):
    def u1(x1, x2):
        return 0.11*x1 + 0.12*x2 + 0.13*x1*x2

    def u2(x1, x2):
        return 0.21*x1 + 0.22*x2 + 0.23*x1*x2

    # Calculate u1 and u2 on the grid
    u1_vals = u1(x1, x2)
    u2_vals = u2(x1, x2)

    # Compute gradients
    eps1_x1 = torch.autograd.grad(u1_vals, x1, torch.ones_like(u1_vals), create_graph=True)[0]
    eps1_x2 = torch.autograd.grad(u1_vals, x2, torch.ones_like(u1_vals), create_graph=True)[0]

    eps2_x1 = torch.autograd.grad(u2_vals, x1, torch.ones_like(u2_vals), create_graph=True)[0]
    eps2_x2 = torch.autograd.grad(u2_vals, x2, torch.ones_like(u2_vals), create_graph=True)[0]

    # Strain components
    eps1 = eps1_x1
    eps2 = eps2_x2
    gam12 = eps1_x2 + eps2_x1

    eps = [eps1.detach().numpy(), eps2.detach().numpy(), gam12.detach().numpy()]
    print("Autograd Strains:")
    for y in eps:
        print(y)

    sig1 = E/(1+v) * (eps1 + v/(1-2*v)*(eps1+eps2))
    sig2 = E/(1+v) * (eps2 + v/(1-2*v)*(eps1+eps2))
    sig3 = E*v/(1+v)/(1-2*v) * (eps1 + eps2)
    tau12 = 0.5*E/(1+v) * gam12

    sig = [sig1.detach().numpy(), sig2.detach().numpy(), tau12.detach().numpy()]
    print("Autograd Stresses:")
    for y in sig:
        print(y)

    sig1_x1 = torch.autograd.grad(sig1, x1, torch.ones_like(sig1), create_graph=True)[0]
    sig2_x2 = torch.autograd.grad(sig2, x2, torch.ones_like(sig2), create_graph=True)[0]
    tau12_x1 = torch.autograd.grad(tau12, x1, torch.ones_like(tau12), create_graph=True)[0]
    tau12_x2 = torch.autograd.grad(tau12, x2, torch.ones_like(tau12), create_graph=True)[0]

    div = [sig1_x1.detach().numpy(), sig2_x2.detach().numpy(), tau12_x1.detach().numpy(), tau12_x2.detach().numpy()]
    print("Autograd div:")
    for y in div:
        print(y)

    b1 = -sig1_x1 - tau12_x2
    b2 = -sig2_x2 - tau12_x1
    return [b1, b2]

    

# Define grid for x1 and x2
x1_vals = torch.linspace(0, 2, 3, requires_grad=True)
x2_vals = torch.linspace(0, 1, 2, requires_grad=True)
x1, x2 = torch.meshgrid(x1_vals, x2_vals, indexing='ij')

banal = analytical(x1, x2)
bauto = autograd(x1, x2)

print("Body forces:")
print(f"Analytical: {banal}")
print(f"Autograd: {bauto}")
