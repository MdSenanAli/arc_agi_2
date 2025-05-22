from manager import Manager
from environment import Environment
from visualizer import Visualizer


if __name__ == "__main__":
    manager = Manager(1, "train")
    i_, o_ = manager.get_next_data()

    env = Environment(i_, o_)
    visual = Visualizer(env)

    visual.run_q_network()
