import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss, Conv1d, Dropout, BatchNorm1d
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# STATISTIC POOLING
# -------------------------
class StatsPooling(torch.nn.Module):
    def forward(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)

# -------------------------
# DATA
# -------------------------
dataset_path = "./data/dataset.pt"

alpha = 0.01
batch_size = 32
epochs = 100
device = "cuda"

data = torch.load(dataset_path)

X = data["X"]   # [N, 80, T]
y = data["y"]

# shuffle (ВАЖНО: без torch.tensor!)
indices = torch.randperm(len(y))

X = X[indices]
y = y[indices]

# train / val
X_train, X_val = X[:3800], X[3800:]
y_train, y_val = y[:3800], y[3800:]

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# -------------------------
# MODEL (FIXED)
# -------------------------
class SpeakerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = Sequential(
            Conv1d(80, 128, kernel_size=5, padding=2),
            ReLU(),
            BatchNorm1d(128),

            Conv1d(128, 128, kernel_size=3, padding=1),
            ReLU(),
            BatchNorm1d(128),

            Conv1d(128, 128, kernel_size=3, padding=1),
            ReLU(),
            BatchNorm1d(128),
        )

        self.pool = StatsPooling()

        self.classifier = Sequential(
            Linear(256, 128),
            ReLU(),
            Dropout(0.3),

            Linear(128, 64),
            ReLU(),

            Linear(64, 10)
        )

    def forward(self, x):
        # x: [B, 80, T]
        x = self.conv(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

model = SpeakerModel().to(device)

# -------------------------
# TRAINING
# -------------------------
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=alpha)

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # safety check (если вдруг перепутан формат)
        if batch_X.shape[1] != 80:
            batch_X = batch_X.transpose(1, 2)

        y_hat = model(batch_X)
        loss = loss_fn(y_hat, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch == 50 or epoch == 70:
        alpha /= 10
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.6f}")

# -------------------------
# VALIDATION
# -------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in val_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        if batch_X.shape[1] != 80:
            batch_X = batch_X.transpose(1, 2)

        y_hat = model(batch_X)
        pred = y_hat.argmax(dim=1)

        correct += (pred == batch_y).sum().item()
        total += len(batch_y)

accuracy = correct / total * 100

print()
print(f"Validation accuracy: {accuracy:.2f}%")