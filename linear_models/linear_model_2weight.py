import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(0)
np.random.seed(0)


# -----------------------------------------------------------------------------#
# Define dataset
# -----------------------------------------------------------------------------#
class ScalarDataset(torch.utils.data.Dataset):
    def __init__(self, num_datasets, transform=None, target_transform=None):
        self._x = np.zeros((num_datasets, 2), dtype=np.float32)

        for i in range(2):
            self._x[:, i] = np.float32(np.random.random([num_datasets]))

        self._y = np.zeros((num_datasets), dtype=np.float32)
        self._y[0] = self._x[0, 0] + 0.5 * self._x[0, 1]
        self._y[1] = 0.7 * self._x[1, 0] + 0.3 * self._x[1, 1]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self._x[:, 0])

    def __getitem__(self, idx):
        x = self._x[idx, :]
        y = self._y[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
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
        return self._linear1(x)

    def print(self):
        print(self._linear1.weight.data)
        print(self._linear1.bias)


# -----------------------------------------------------------------------------#
# Define training function:
# -----------------------------------------------------------------------------#
def train(dataloader, model, loss_fn, optimizer, scheduler, iter_tol=1e-6):
    size = len(dataloader.dataset)
    model.train()
    loss_values = []
    weights_all = []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        weights = model.state_dict()
        weights_all.append(np.array(weights["_linear1.weight"]))
        loss_value, current = loss.item(), (batch + 1) * len(X)
        loss_values.append(loss_value)

    #    scheduler.step()

    return loss_values, weights_all


batch_size = 1
num_epochs = 100
num_datasets = 2

training_data = ScalarDataset(num_datasets)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

model = NeuralNetwork().to(device)
loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.RMSprop(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
# optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.AdamW(model.parameters())


# optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#    optimizer,
#    max_lr=1.0,
#    steps_per_epoch=len(training_data),
#    epochs=num_epochs,
#    verbose=False,
#    final_div_factor=1e4,
#    anneal_strategy="cos",
# )
scheduler = []


# --------------------------------------------------------------------------------#
# Training loop
# --------------------------------------------------------------------------------#
iter_tol = 1e-10
loss_prev = 1.0
lossall = []
weights_all = []
for epoch in range(num_epochs):
    loss, weights = train(train_dataloader, model, loss_fn, optimizer, scheduler)

    print(f"iter: {epoch:d}, loss: {loss[-1]:>7e}")

    if np.abs(loss[-1] - loss_prev) < iter_tol:
        break

    loss_prev = loss[-1]
    lossall.extend(loss)
    weights_all.extend(weights)

model.print()

weights_all = np.squeeze(np.array(weights_all))


# --------------------------------------------------------------------------------#
# Plot loss
# --------------------------------------------------------------------------------#
plt.semilogy(lossall, ".-")
plt.grid(True)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the 3D polyline
print(weights_all.shape)
ax.plot(weights_all[:, 0], weights_all[:, 1], lossall, ".-")

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
