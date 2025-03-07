import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)


A = np.loadtxt('linear_advection_features.csv', delimiter=',')
b   = np.loadtxt('linear_advection_labels.csv',   delimiter=',')

lhs = np.dot(A.T, A)
rhs = np.dot(A.T, b)

w = np.linalg.solve(lhs, rhs)

print(w)
