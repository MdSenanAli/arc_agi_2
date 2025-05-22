# 🧠 Solving ARC AGI 2 with Deep Q-Learning

This project implements a **Deep Q-Learning** (DQN) agent designed to solve challenges from the [ARC AGI 2](https://github.com/fchollet/ARC) dataset—an artificial intelligence benchmark that tests human-like reasoning and abstraction through grid-based tasks. Inspired by reinforcement learning approaches in classic games like **Pac-Man**, the agent learns how to transform input grids to desired output patterns through trial and reward.

---

## 🎯 Project Goal

The aim is to explore how reinforcement learning, particularly Deep Q-Learning, can be applied to **abstract visual reasoning tasks**. Just like in **Pac-Man**, where an agent must learn to survive and maximize score through feedback from its environment, our agent navigates the search space of transformations and learns optimal strategies through rewards.

---

## 🧩 What is ARC AGI 2?

The **Abstraction and Reasoning Corpus (ARC)** by François Chollet is a benchmark designed to measure how well a system can generalize from a few examples—something humans do effortlessly.  
Each task in ARC presents a few **input-output grid pairs**, and the system must learn the transformation rule and apply it to a new, unseen input.

---

## 🚀 Approach

### Reinforcement Learning Pipeline

- **Environment**: Modeled each ARC task as a custom environment where the agent’s goal is to convert the input grid to match the output.
- **States**: Represent the current grid configuration.
- **Actions**: Defined as transformations (e.g., color changes, shape movements).
- **Rewards**: Positive when the transformation brings the grid closer to the correct output, negative otherwise.
- **Agent**: A Deep Q-Network (DQN) that learns which transformation actions maximize long-term rewards.

This mirrors how agents learn in game environments—**taking sequential actions to maximize future rewards**.

---

## 🛠️ Tech Stack

- Python
- NumPy
- PyTorch
- PyGame
---

## 📈 Reinforcement Learning vs. Traditional Approaches

| Feature                | Traditional Solvers | DQN Agent (This Project)     |
|------------------------|---------------------|------------------------------|
| Learning from reward   | ❌ No                | ✅ Yes                        |
| Generalization         | ⚠️ Limited           | ✅ Emerging generalization   |
| Human-like trial/error | ❌ Mostly symbolic   | ✅ Trial-and-error based     |
| Game-like behavior     | ❌                  | ✅ Pac-Man-style exploration |

---
