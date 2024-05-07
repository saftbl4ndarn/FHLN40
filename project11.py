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

    