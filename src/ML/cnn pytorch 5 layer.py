import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/total dataset for ML with statistic numbers.xlsx'
data = pd.read_excel(file_path)

# Extract features and labels
X = data.iloc[:, :96].values  # first 96 columns as features
y = data.iloc[:, 106].values  # assuming 97th column as labels and zero-indexing correction

# Reshape X to have each instance as 48x2 (48 points with x, y coordinates)
X = X.reshape(-1, 48, 2)

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(4, 1), stride=(2, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(2, 1), stride=1)  # New layer

        self._to_linear = None
        self._get_conv_output_size([1, 48, 2])

        self.fc = nn.Linear(self._to_linear, 5)  # Assuming 5 is the number of classes

    def _get_conv_output_size(self, shape):
        input = torch.rand(*shape)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)  # Pass through the new layer
        self._to_linear = int(np.prod(output.size()))

    def forward(self, x):
        x = x.view(-1, 1, 48, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))  # Pass through the new layer
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = CNN()
print(model)
valloss = []
acc = []
# Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        print(f'Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}, Accuracy: {100. * correct / len(val_loader.dataset)}%')
        valloss.append(val_loss / len(val_loader))
        acc.append(correct / len(val_loader.dataset))

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=2000)

# Testing and Evaluation
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.numpy())
        test_targets.extend(labels.numpy())

conf_matrix = confusion_matrix(test_targets, test_preds)
accuracy = accuracy_score(test_targets, test_preds)
df = pd.DataFrame({
    'validation loss': valloss,
    'accuracy': acc
})
df.to_excel('5 layer CNN total dataset validation loss and accuracy lr = 0.00005.xlsx', index=None)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.title(f'5 layers CNN Confusion Matrix Accuracy = {round(accuracy,4)}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# print(f'Confusion Matrix:\n{conf_matrix}')
# print(f'Accuracy: {accuracy * 100:.2f}%')
