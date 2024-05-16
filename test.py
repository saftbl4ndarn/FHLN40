def autodiff(x1, x2, u1_vals, u2_vals):
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

    sig1 = E/(1+v) * (eps1 + v/(1-2*v)*(eps1+eps2))
    sig2 = E/(1+v) * (eps2 + v/(1-2*v)*(eps1+eps2))
    sig3 = E*v/(1+v)/(1-2*v) * (eps1 + eps2)
    tau12 = 0.5*E/(1+v) * gam12

    sig = [sig1.detach().numpy(), sig2.detach().numpy(), tau12.detach().numpy()]
    
    sig1_x1 = torch.autograd.grad(sig1, x1, torch.ones_like(sig1), create_graph=True)[0]
    sig2_x2 = torch.autograd.grad(sig2, x2, torch.ones_like(sig2), create_graph=True)[0]
    tau12_x1 = torch.autograd.grad(tau12, x1, torch.ones_like(tau12), create_graph=True)[0]
    tau12_x2 = torch.autograd.grad(tau12, x2, torch.ones_like(tau12), create_graph=True)[0]

    b1 = -sig1_x1 - tau12_x2
    b2 = -sig2_x2 - tau12_x1
    return [b1, b2], eps, sig