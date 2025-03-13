import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
np.set_printoptions(precision=16)
torch.set_default_dtype(torch.float64)


# -----------------------------------------------------------------------------#
# Define dataset
# -----------------------------------------------------------------------------#
class ScalarDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        features = np.loadtxt("linear_advection_features.csv", delimiter=",")
        labels = np.loadtxt("linear_advection_labels.csv", delimiter=",")

        features = features.astype(np.float64)
        labels = labels.astype(np.float64)

        self._x = features
        self._y = np.zeros((len(labels), 1), dtype=np.float64)
        self._y[:, 0] = labels

        print("#features: " + str(features.shape))
        print("#labels:   " + str(labels.shape))

    def __len__(self):
        return len(self._x[:, 0])

    def __getitem__(self, idx):
        x = self._x[idx, :]
        y = self._y[idx]
        return x, y


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# -----------------------------------------------------------------------------#
# Define model:
# -----------------------------------------------------------------------------#
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._linear1 = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x):
        x1 = self._linear1(x)
        return x1

    def print(self):
        print(self._linear1.weight.data)


# -----------------------------------------------------------------------------#
# Define training function:
# -----------------------------------------------------------------------------#
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    loss_values = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

    return loss_values


batch_size = 1
num_epochs = 200

training_data       = ScalarDataset()
train_dataloader    = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

model       = NeuralNetwork().to(device)
loss_fn     = torch.nn.MSELoss()
optimizer   = torch.optim.SGD(model.parameters(), lr=0.5)


# --------------------------------------------------------------------------------#
# Training loop
# --------------------------------------------------------------------------------#
tol = 1e-10
rmse = []

for epoch in range(num_epochs):

    loss = train(train_dataloader, model, loss_fn, optimizer)
    
    rmse.append(np.sqrt(np.mean(loss)))

    print(f"epoch: {epoch:d}, loss: {rmse[-1]:>7e}")

    if np.abs(rmse[-1]) < tol:
        break

model.print()
print(np.array(model._linear1.weight.data).squeeze())

# --------------------------------------------------------------------------------#
# Plot loss
# --------------------------------------------------------------------------------#
plt.plot(np.log10(rmse), ".-")
plt.xlabel("Epoch")
plt.ylabel("log10(rmse)")
plt.title("Training loss")
plt.ylim(-16, 0)
plt.grid(True)
plt.show()
