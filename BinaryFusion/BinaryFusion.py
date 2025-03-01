import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

class BinaryFusion(nn.Module):
    def __init__(self):
        super(BinaryFusion, self).__init__()
        # Initialize parameters a and b (corresponding to weights [a, a] and [b, b])
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        """
        Forward pass:
        x: Tensor of shape (batch_size, 2)
           Each row contains [x1, x2], where x1 and x2 are the outputs from two models (0 or 1)
        """
        # Compute the sum of the two inputs: s = x1 + x2 (shape: (batch_size, 1))
        s = torch.sum(x, dim=1, keepdim=True)
        # Calculate logits for the two models
        logit1 = self.a * s  # for model 1
        logit2 = self.b * s  # for model 2
        # Concatenate the logits to form a (batch_size, 2) tensor
        logits = torch.cat([logit1, logit2], dim=1)
        # Apply softmax to obtain probabilities
        probs = F.softmax(logits, dim=1)
        return probs

# Create model, optimizer, and loss function
model = BinaryFusion()
optimizer = optim.Adam(model.parameters(), lr=55)
criterion = nn.CrossEntropyLoss()

# Load the training data from CSV (expected columns: x1, x2, label)
train_data = pd.read_csv('train_data.csv')
X_train = torch.tensor(train_data[['x1', 'x2']].values, dtype=torch.float32)
y_train = torch.tensor(train_data['label'].values, dtype=torch.long)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # Forward pass: compute probabilities
    probs = model(X_train)
    # Compute cross-entropy loss
    loss = criterion(probs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Inference phase: compute predictions and calculate accuracy
with torch.no_grad():
    probs = model(X_train)
    # Predict: 0 means choose model 1, 1 means choose model 2
    preds = torch.argmax(probs, dim=1)
    # Print the predicted tensor
    print("Predicted tensor:", preds)
    # Calculate and print accuracy
    accuracy = (preds == y_train).sum().item() / y_train.size(0)
    print("Accuracy:", accuracy)
    # Append predictions as a new column to the training DataFrame
    train_data['predictions'] = preds.numpy()

# Save the updated DataFrame back to a CSV file
train_data.to_csv('train_data_with_predictions.csv', index=False)