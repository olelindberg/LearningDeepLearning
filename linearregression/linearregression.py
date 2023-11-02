import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


num_iter = 1000
iter_tol = 1e-6

x = np.array([[1], [2], [3], [3.5]])
y = np.array([[2], [4], [6], [7.5]])


# instantiate model class
input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1.0,
    steps_per_epoch=1,
    epochs=num_iter)

loss_prev = 0.0
for iter in range(num_iter):
    inputs = Variable(torch.from_numpy(x))
    labels = Variable(torch.from_numpy(y))  # clear gradients wrt parameters

    outputs = model(inputs.float())

    loss = criterion(outputs.float(), labels.float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    print("iter {}, loss {}".format(iter, loss.data))

    if (np.abs(loss.data-loss_prev)<iter_tol):
        break
    loss_prev = loss.data


predicted = model.linear(Variable(torch.from_numpy(x)).float())

# plot graph
import matplotlib.pyplot as plt

# clear figure
plt.clf()  # get predictions
predicted = predicted  # plot true data
plt.plot(x, y, "go", label="True data", alpha=0.50)  # plot predictions
plt.plot(x, predicted.detach().numpy(), "-", label="Predictions", alpha=0.50)  # Legend and plot
plt.legend(loc="best")
plt.show()
