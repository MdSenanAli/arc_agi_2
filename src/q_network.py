import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(
        self, input_dim=2 * 63 + 2, output_dim=14, learning_rate=1e-3, gamma=0.99
    ):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def update(self, state, action, reward, next_state, done, target_network):
        current_q_values = self(state)
        current_q_value = current_q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = target_network(next_state)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_value = reward + (self.gamma * max_next_q_values * (1 - done))

        loss = F.mse_loss(current_q_value, target_q_value)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
