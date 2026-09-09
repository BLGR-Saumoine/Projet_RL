# Deep Reinforcement Learning: Atari Agent from Scratch

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) 
![Python](https://img.shields.io/badge/Python-3.10%2B-blue) 
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![Gymnasium](https://img.shields.io/badge/Gymnasium-ALE-lightgrey)

Academic project by Gendronneau Maël.

## Project Overview

This project aims to implement advanced Deep Q-Learning algorithms from scratch using PyTorch to train an autonomous agent to play Atari 2600 games (primarily Pong and Qbert). The objective is to move beyond the vanilla Deep Q-Network (DQN) by integrating modern improvements from the "Rainbow" architecture, such as Dueling Networks, Prioritized Experience Replay (PER), and N-step returns, directly learning from raw pixel inputs.

## Repository Structure

* **reinforcementLearningMain.ipynb**: Main notebook containing the training loops, environment setup, experiment tracking, and performance visualizations (moving averages).
* **agent.py**: Core implementation of the RL Agent, handling action selection (epsilon-greedy), loss computation (Smooth L1 Loss), and soft/hard network updates.
* **dqnModel.py**: PyTorch neural network architectures, including the Convolutional Neural Network (CNN) feature extractor and the Dueling DQN streams.
* **memories.py**: Custom replay buffer implementations, including a standard Replay Memory, an N-step Small Memory, and a Prioritized Experience Replay (PER) buffer using proportional prioritization.
* **scores_*.csv / *.pth**: Saved model weights and training logs for performance analysis.

## Key Components Implemented

### Core Architecture
- **Convolutional Feature Extractor**: Processes stacked grayscale frames (84x84) to extract spatial and temporal features.
- **Vanilla DQN**: Baseline Q-value estimation.
- **Dueling DQN**: Separate streams for State-Value $V(s)$ and Advantage $A(s, a)$ estimation to improve learning stability across similar actions.

### Architectural Elements & RL Techniques
- **Prioritized Experience Replay (PER)**: Samples transitions based on the magnitude of their Temporal Difference (TD) error, accelerating learning on "surprising" transitions.
- **Multi-Step Learning (N-step)**: Bootstraps the target Q-value over 3 steps to propagate delayed rewards faster.
- **Double DQN (DDQN)**: Decouples action selection from action evaluation to reduce overestimation bias.
- **Frame Stacking**: Stacks 4 consecutive frames to provide the agent with the concept of velocity and direction.

## Methodology

### Environment (Gymnasium ALE)
- **Games**: `ALE/Pong-v5` (and Qbert).
- **Preprocessing**: 
   - Grayscale conversion
   - Resizing to 84x84 pixels
   - Frame Stacking (4 frames)
   - Reward clipping

### Experiments Conducted
1. **Baseline Comparison**: Vanilla DQN vs. Dueling DQN vs. combined advanced architectures.
2. **Buffer Analysis**: Standard Replay Buffer vs. Prioritized Experience Replay (PER).
3. **Reward Propagation**: The impact of 1-step vs. 3-step returns on early-stage learning speed and stability.

### Implementation Details
- Pure PyTorch implementation with customized training loops.
- Moving average visualizations to track the agent's transition from pure exploration (random play/losing) to defensive mastery, and finally to offensive winning strategies.

## Results & Insights

The experiments demonstrate clear evidence of the agent's learning phases:
- **Phase 1 (Exploration)**: High loss, short survival times.
- **Phase 2 (Defense)**: Significant increase in game length (number of steps) as the agent learns to survive without necessarily scoring.
- **Phase 3 (Mastery)**: Rapid score improvement leading to near-perfect games (e.g., +17 to +21 scores in Pong), heavily accelerated by the combination of Dueling DQN, PER, and 3-step returns.

<div align="center">
<figure>
  <video src="https://github.com/BLGR-Saumoine/Projet_RL/blob/main/double3step_PONG.mp4" controls width="50%"></video>
  <figcaption><em>Average performances on PONG (Green)</em></figcaption>
</figure>
</div>

## Key References
- Mnih et al. (2015) - Human-level control through deep reinforcement learning
- Wang et al. (2015) - Dueling Network Architectures for Deep Reinforcement Learning
- Schaul et al. (2015) - Prioritized Experience Replay
- Hessel et al. (2017) - Rainbow: Combining Improvements in Deep Reinforcement Learning

## Authors
- Gendronneau Maël

*This project was completed as part of an academic curriculum in deep learning and reinforcement learning.*
