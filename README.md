# 🚦 EmergeRL — Reinforcement Learning Traffic Signal Control

A Python-based traffic intersection simulation that explores how reinforcement learning can be used to control traffic signals, manage vehicle queues, and prioritize emergency vehicles.

Built as a group project for **COMP 4010: Introduction to Reinforcement Learning** at Carleton University.

## 📌 Project Overview

EmergeRL simulates a four-way traffic intersection with traffic signals, turning lanes, incoming vehicles, and emergency vehicles.

The project uses a custom **Gymnasium environment** to model traffic conditions and train reinforcement learning agents. A **Pygame visualization** displays the intersection, vehicle movement, and traffic-light changes in real time.

The project explores two reinforcement learning approaches:

- **SARSA:** A tabular reinforcement learning algorithm that learns traffic signal decisions through rewards and penalties.
- **PPO-based Agent (MAPPO implementation):** A PyTorch-based policy optimization agent that learns traffic signal actions using actor and critic networks.

## ✨ Features

- 🚗 Four-way traffic intersection with straight and left-turning vehicles
- 🚦 Programmable traffic signals and multiple signal configurations
- 🚑 Emergency vehicle detection and priority-related rewards
- 🧠 SARSA and PPO-based reinforcement learning agents
- 📊 Performance tracking for vehicle queues, waiting time, cumulative rewards, and emergency response
- 🎮 Real-time traffic visualization using Pygame
- ⚙️ Custom Gymnasium environment with configurable traffic arrival rates

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Gymnasium | Traffic simulation environment |
| Pygame | Interactive visualization |
| PyTorch | PPO actor-critic neural networks |
| NumPy | Numerical operations and state processing |
| SARSA / PPO | Reinforcement learning algorithms |

## 📂 Project Structure

```text
RL-Traffic-Signal-Control/
│
├── assets/
│   ├── car.png
│   └── emergency.png
│
├── gymnasium_env/
│   ├── __init__.py
│   └── envs/
│       ├── __init__.py
│       └── traffic_world.py
│
├── Project_Progress_Updates/
│
├── sarsa_agent.py
├── mappo_agent.py
├── pygame_test.py
├── run_gymnasium_env.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/LavenGit/RL-Traffic-Signal-Control.git
cd RL-Traffic-Signal-Control
```

### 2. Install Dependencies

Python 3.10 was used for the original project.

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Pygame Simulation

```bash
python pygame_test.py
```

The Pygame window displays the traffic intersection and vehicle movements.

To switch between reinforcement learning agents, open `pygame_test.py` and change:

```python
agent_type = "MAPPO"
```

to:

```python
agent_type = "SARSA"
```

### 4. Run the Gymnasium Environment

To test the traffic environment separately in the terminal:

```bash
python run_gymnasium_env.py
```

This runs the environment with randomly selected actions and prints observations, rewards, and traffic information.

## 📈 Reinforcement Learning Approach

The agents interact with the traffic environment through observations, actions, and rewards.

**Observations:** Traffic-light states, vehicle queues, emergency vehicle indicators, and simulation time.

**Actions:** Traffic signal control decisions.

**Rewards and penalties:** Vehicle throughput, waiting queues, signal switching, and emergency vehicle handling.

The project is an experimental implementation. Agent performance and training behavior are still being refined, and improvements over fixed-timing traffic control have not yet been established through systematic evaluation.

## 👥 Project Team

**Carleton University — Fall 2025**

**Course:** COMP 4010 — Introduction to Reinforcement Learning

**Instructor:** Junfeng Wen

**Team Members:**
- Nikhil Bansal
- Sham Ruthirapragas
- Rohan Doulakar
- Laven Joshi
- Bibek Singh Kahlon

## 📄 License

See the [LICENSE](LICENSE) file for license information.
