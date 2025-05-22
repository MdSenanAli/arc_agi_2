import pygame
from constants import COLOR_MAP
import random
from q_network import QNetwork
import torch

# Convert hex color codes to RGB tuples
PYGAME_COLOR_MAP = {key: pygame.Color(value) for key, value in COLOR_MAP.items()}


class Visualizer:
    def __init__(self, environment):
        self.env = environment
        self.rows = environment.rows
        self.cols = environment.cols

        self.window_size = [800, 500]
        self.grid_spacing = 40
        self.border_padding = 40
        self.border_thickness = 2

        available_width = (
            self.window_size[0] - 2 * self.border_padding - self.grid_spacing
        )
        self.cell_size = min(
            available_width // (2 * self.cols),
            self.window_size[1] - 2 * self.border_padding // self.rows,
            self.window_size[0] - 2 * self.border_padding // self.cols,
        )

        self.window_size[0] += 10
        self.window_size[1] += 10
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption(
            "ARC AGI - Visualization Environment (Made By Senan)"
        )

        # Icon
        icon = pygame.image.load("assets\icon.png")
        pygame.display.set_icon(icon)

        # Qnet
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.episodes = 1
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.update_target_every = 5
        self.iter = 0
        self.max_iter = 100
        self.total_reward = 0

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        self.env.step(self.env.index_to_action.index("up"))
                    elif event.key == pygame.K_DOWN:
                        self.env.step(self.env.index_to_action.index("down"))
                    elif event.key == pygame.K_LEFT:
                        self.env.step(self.env.index_to_action.index("left"))
                    elif event.key == pygame.K_RIGHT:
                        self.env.step(self.env.index_to_action.index("right"))
                    elif pygame.K_0 <= event.key <= pygame.K_9:
                        color_index = event.key - pygame.K_0
                        self.env.step(color_index)

            self.update_display()

        pygame.quit()

    def run_random(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            number = random.randint(0, 13)
            reward, terminal = self.env.step(number)
            if terminal:
                running = False

            self.update_display()

        pygame.quit()

    def run_q_network(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            terminal = self.q_network_helper()
            if terminal:
                running = False
            self.update_display()

        pygame.quit()

    def q_network_helper(self):
        state = self.env.get_state_vector()
        print(self.epsilon)
        if random.random() < self.epsilon:
            action = random.randint(0, 13)  # 14 actions
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax().item()

        reward, done = self.env.step(action)
        next_state = self.env.get_state_vector()

        loss = self.q_network.update(
            state,
            torch.tensor([action]),
            torch.tensor([reward], dtype=torch.float32),
            next_state,
            torch.tensor([done], dtype=torch.float32),
            self.target_network,
        )

        self.total_reward += reward
        self.iter += 1
        if self.iter % self.max_iter == 0:
            self.epsilon *= self.epsilon_decay
        print(self.total_reward)
        return done

    def update_display(self):
        self.screen.fill(pygame.Color(30, 30, 30))

        total_grid_width = 2 * self.cols * self.cell_size + self.grid_spacing
        start_x = (self.window_size[0] - total_grid_width) // 2
        start_y = (self.window_size[1] - self.rows * self.cell_size) // 2

        # Draw current state grid
        self.draw_grid(self.env.curr_state, offset_x=start_x, offset_y=start_y)
        self.draw_border(start_x, start_y)

        # Draw target grid
        target_x = start_x + self.cols * self.cell_size + self.grid_spacing
        self.draw_grid(self.env.target, offset_x=target_x, offset_y=start_y)
        self.draw_border(target_x, start_y)

        # Draw agent
        self.draw_agent(self.env.agent_pos, offset_x=start_x, offset_y=start_y)

        pygame.display.flip()

    def draw_grid(self, grid, offset_x=0, offset_y=0):
        for row in range(self.rows):
            for col in range(self.cols):
                cell_value = int(grid[row][col])
                color = PYGAME_COLOR_MAP.get(cell_value, pygame.Color(0, 0, 0))
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        offset_x + col * self.cell_size,
                        offset_y + row * self.cell_size,
                        self.cell_size - 1,
                        self.cell_size - 1,
                    ),
                )

    def draw_border(self, x, y):
        width = self.cols * self.cell_size
        height = self.rows * self.cell_size
        pygame.draw.rect(
            self.screen,
            pygame.Color(200, 200, 200),
            (
                x - self.border_thickness,
                y - self.border_thickness,
                width + 2 * self.border_thickness,
                height + 2 * self.border_thickness,
            ),
            self.border_thickness,
        )

    def draw_agent(self, pos, offset_x=0, offset_y=0):
        row, col = pos
        center_x = offset_x + col * self.cell_size + self.cell_size // 2
        center_y = offset_y + row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 4

        pygame.draw.circle(
            self.screen,
            pygame.Color("white"),
            (center_x, center_y),
            radius,
        )
