import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
np.set_printoptions(precision=16)

A = np.loadtxt('linear_advection_features_upwind.csv', delimiter=',')
b   = np.loadtxt('linear_advection_labels_upwind.csv',   delimiter=',')

lhs = np.dot(A.T, A)
rhs = np.dot(A.T, b)

w = np.linalg.solve(lhs, rhs)


print(w)