import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleQ(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(SimpleQ, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_actions)
        self.dropout = nn.Dropout(0.5)
        
        # Use Xavier uniform distribution to initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def Q_construct(input_dim, num_actions):
    return SimpleQ(input_dim=input_dim, num_actions=num_actions)

if __name__ == "__main__":
    # Example usage
    batch_size = 8
    input_dim = 128
    num_actions = 5

    # Create input tensor
    x = torch.randn(batch_size, input_dim)

    # Create neural network instance
    model = Q_construct(input_dim=input_dim, num_actions=num_actions)

    # Perform forward pass
    output = model(x)

    # Print output
    print("Output:", output)
