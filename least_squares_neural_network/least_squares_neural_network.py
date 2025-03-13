import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data (for example)
torch.manual_seed(42)
X = torch.randn(100, 3)  # 100 samples, 3 features
true_W = torch.tensor([[2.0], [-3.0], [4.0]])  # True weights
Y = X @ true_W + 0.1 * torch.randn(100, 1)  # Adding some noise


# Define a simple neural network
class LeastSquaresNN(nn.Module):
    def __init__(self, input_dim):
        super(LeastSquaresNN, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)  # No bias to mimic least squares

    def forward(self, x):
        return self.linear(x)


# Initialize model, loss, and optimizer
model = LeastSquaresNN(input_dim=3)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the network
loss_all = []
num_epochs = 64
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = loss_fn(predictions, Y)
    loss.backward()
    optimizer.step()

    loss_all.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Print the learned weights
print("Learned weights:", model.linear.weight.data)
print("True weights:", true_W.T)


plt.semilogy(np.array(loss_all) / loss_all[0], "-o")
plt.grid()
plt.show()
