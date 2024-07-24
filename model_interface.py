#later import the xai assistant which has these functions
# def predict(x1,x2,x3):
#     return "57"

# def explain(x1,x2,x3):
#     return "importance of x1 = 0.65, x2 = 0.25, x3 = 0.10"


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from icecream import ic

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class PredictionModel:
    def __init__(self, learning_rate=0.01):
        self.model = SimpleNN()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self._is_trained = False
    def __call__(self, data):
        print("Calling the model")
        self.model.eval()
        # Convert the input data to a PyTorch tensor if it's not already
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            data = data.float()
        
        # Ensure the model outputs a tensor with the shape (num_samples, 1)
        with torch.no_grad():
            return self.model(data).numpy()
    def generate_data(self,num_samples=100):
        np.random.seed(42)
        X = np.random.rand(num_samples, 3)
        y = (np.sum(X, axis=1) > 1.5).astype(int)  # boolean label based on a simple rule
        self.X = X
        self.y = y
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        return X_tensor, y_tensor
    def train(self, num_epochs=100):
        X, y = self.generate_data()

        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        self._is_trained = True
    def predict(self, x1, x2, x3):
        if not self._is_trained:
            print("Model is not trained yet. Going to train model.")
            self.train()
        self.model.eval()
        X = torch.tensor([[x1, x2, x3]], dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X)
            predicted_labels = (predictions > 0.5).float()
        return predicted_labels

    def save_model(self, path="model.pth"):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path="model.pth"):
        self.model.load_state_dict(torch.load(path))
        self._is_trained = True
        return self
