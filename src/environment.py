import torch
from spp_layer import SPPLayer


class Environment:
    def __init__(self, input_t, output_t, start=(0, 0)):
        self.input_t = input_t.clone()
        self.spp = SPPLayer(levels=[1, 2, 4], pool_types=["max", "avg", "min"])
        self.input_flat = self.spp.forward(input_t)
        self.target = output_t.clone()
        self.rows, self.cols = output_t.shape

        self.start = start
        self.agent_pos = start
        self.curr_state = torch.zeros(output_t.shape)
        self.colored_state = torch.zeros(output_t.shape)

        self.index_to_action = [f"color_{i}" for i in range(10)] + [
            "up",
            "down",
            "left",
            "right",
        ]
        self.last_color = None
        self.last_location = None

    def is_terminal(self):
        return torch.equal(self.curr_state, self.target)

    def step(self, action_index):
        action = self.index_to_action[action_index]
        row, col = self.agent_pos
        reward = 0

        move_loss = 10

        if action == "up":
            row = max(row - 1, 0)
            reward -= move_loss
        elif action == "down":
            row = min(row + 1, self.rows - 1)
            reward -= move_loss
        elif action == "left":
            col = max(col - 1, 0)
            reward -= move_loss
        elif action == "right":
            col = min(col + 1, self.cols - 1)
            reward -= move_loss
        elif action.startswith("color_"):
            color_value = int(action.split("_")[1])
            self.curr_state[row][col] = color_value

            if color_value == self.target[row, col]:
                reward = 1  # correct color
            else:
                reward = -5  # wrong color

            if self.colored_state[row, col]:
                reward = -100
            self.colored_state[row, col] = 1

        diff = (self.curr_state - self.target).abs()
        closeness_score = (diff == 0).float().sum()  # Number of correctly colored cells
        reward += closeness_score.item()

        terminal = self.is_terminal()
        if terminal:
            reward = 10000000
        self.agent_pos = (row, col)

        return reward, terminal

    def get_state_vector(self):
        current_flat = self.spp.forward(self.curr_state)
        return torch.cat(
            [
                self.input_flat,
                current_flat,
                torch.tensor(self.agent_pos, dtype=torch.float32).view(1, -1),
            ],
            dim=1,
        )
