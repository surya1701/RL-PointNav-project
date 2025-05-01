---
title: PointNav Agents in RoboTHOR
layout: default
---

# PointNav Agents in RoboTHOR

### Team Members
- Suryavardan Suresh  
- Twishaa Sahay  
- Dev Pant  
- Atmaj Koppikar

---

## Abstract

In this project, we investigate the problem of Point Navigation (PointNav) in realistic 3D indoor environments using the RoboTHOR simulation platform. The objective is to train an agent to reach a target location using first-person visual input, without access to a map or GPS. We explore a range of navigation strategies, from naive random action selection and rule-based heuristics to learning-based approaches using reinforcement learning (RL) algorithms.

To benchmark and compare performance, we evaluate each agent under the same environment constraints and goal conditions. The agents include:
- A **Random policy**, as a lower-bound baseline.
- A **Heuristic policy** that rotates toward the goal and attempts to move forward.
- A **DQN agent** that learns a Q-value function over discretized actions from visual inputs.
- A **PPO agent**, using an actor-critic framework with advantage-based policy updates.

This task has direct relevance to real-world applications such as indoor robot navigation, warehouse automation, and home-assistant robots. Solving PointNav in simulation is a crucial stepping stone toward deploying navigation agents in physical environments, where access to maps, GPS, or full environment layouts is not guaranteed. The use of visual input and learned policies allows agents to generalize across different layouts and adapt to partial observability, making these methods practical for real deployment.

Our pipeline supports RGB, depth, and RGB+depth observation modes, reward shaping, training with real PointNav datasets from RoboTHOR, and video-based rollout visualization. We track training progression via reward curves and visualize agent behavior through episodic videos. Our results demonstrate that reinforcement learning methods significantly outperform non-learning baselines in both reward accumulation and goal-reaching behavior, with PPO showing improved stability over DQN.


---

## Dataset

### Task Overview: ObjectNav vs PointNav

The AI2-RoboTHOR simulation framework supports two primary embodied AI navigation tasks:

- **Object Navigation (ObjectNav):** Navigate to an object category (e.g., "mug") based on a semantic goal.
- **Point Navigation (PointNav):** Navigate to a specific 3D coordinate provided as a goal, using only egocentric sensory inputs.

Our project focuses on the **PointNav task** using the RoboTHOR environment. While it orginially is a collection of 3D indoor scenes developed for research on real-world navigation and transfer learning. The authors have also provided environments that support navigation around the simulated scene in the Unity3D engine.
The video below is from the creators of the dataset and the RoboTHOR tasks.

<div style="padding:56.25% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/509326657?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="RoboTHOR"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

---

### RoboTHOR Environment and Dataset

We use the [AllenAct](https://github.com/allenai/allenact) implementation of the RoboTHOR PointNav dataset. Each episode specifies:

- A **scene** (e.g., `FloorPlan_Train5_1`)
- The agent's **initial position** and **orientation**
- A **goal position**
- Optionally, the **shortest path** between them for evaluation

#### 🔎 Example Episode (JSON)

```json
{
  "scene": "FloorPlan_Train5_1",
  "initial_position": {"x": 1.5, "y": 0.9, "z": -3.0},
  "initial_orientation": 90.0,
  "target_position": {"x": 2.25, "y": 0.9, "z": -1.75}
}
```

Example of one of the scenes from the dataset, with the green dot denoting the start point and the red dot denoting the end point.

| Scene 1         | Scene 2       |
|-------------------------|------------------------|
|<img width="320"  src="https://github.com/surya1701/RL-PointNav-project/raw/refs/heads/main/assets/harder_top_down_view.png"/>  | <img width="320"  src="https://github.com/surya1701/RL-PointNav-project/raw/refs/heads/main/assets/top_down_view.png"/> |


Please note that the top-level view is just for demonstration, the dataset training is done on the egocentric view of the agent.
A video of the agent's view for the second scene can be viewed in the episode rollout section below with the caption "Couch".

---

### Environment Configuration

The RoboTHOR simulator operates with the following default settings:

| Parameter              | Value                |
|------------------------|----------------------|
| `grid_size`            | 0.25 meters          |
| `rotate_step_degrees` | 90°                  |
| `visibility_distance` | 1.5 meters           |
| `field_of_view`        | 60°                  |
| `movementGaussianSigma` | 0.000001           |
| `rotateGaussianSigma`   | 0.000001           |

These parameters are used to setup an environment that after each action returns the observation as a frame or image. We have a **discrete action space** i.e. the agent can take one of the following actions at each step:

- `MoveAhead`
- `RotateLeft`
- `RotateRight`
- `LookUp`
- `LookDown`

---

### Observation Modalities

We configure the agent to receive **egocentric** observations in one of the following formats:
- **RGB only**
- **Depth only**
- **RGB + Depth** (concatenated as 4 channels)

#### 📷 Example Observations

| Agent RGB View         | Agent Depth Image       |
|-------------------------|------------------------|
|<video width="320" height="240" controls src="https://github.com/surya1701/RL-PointNav-project/raw/refs/heads/main/assets/FloorPlan_Train3_2_random_rgb.mp4" type="video/mp4"></video> | <video width="320" height="240" controls  src="https://github.com/surya1701/RL-PointNav-project/raw/refs/heads/main/assets/FloorPlan_Train3_2_random_depth.mp4"></video> |


---

### Custom Reward Function

To guide the agent effectively during training, we iterated over multiple reward functions. Starting with a simple motion penalty and destination reward to then implementing a custom **reward shaping** mechanism tailored for the PointNav task. We tried to incorporate distance improvement credits, obstacle awareness and behavior discouragement for inefficient or repetitive actions.

---

#### 📐 Reward Components

At each timestep, the reward is computed as:

\[
r = r_{\text{base}} + r_{\text{collision}} + r_{\text{distance}} + r_{\text{loop\_penalty}} + r_{\text{stationary\_penalty}} + r_{\text{depth\_bonus}} + r_{\text{success}}
\]

Where:

| Component            | Value / Condition                                                                 |
|----------------------|-----------------------------------------------------------------------------------|
| **Base step penalty**       | \(-0.01 \) per step                                      |
| **Collision penalty**       | \(-1.0 \) if `action fails`            |
| **Distance shaping**        | \(2 \cdot (\text{prev\_dist} - \text{curr\_dist}) \) |
| **Loop penalty**            | \( -0.2 \) if action is RotateLeft/Right or LookUp/Down     |
| **Stationary penalty**       | \( -0.05 \) if action is Look or Rotate                                     |
| **Depth navigation bonus**  | \( +0.1 \) if `MoveAhead` follows `LookUp` or `LookDown`                   |
| **Success reward**          | \( +10.0 \) if agent is within 0.5m of goal                                 |

---

#### ✅ Reward Conditions (Visual Summary)

```text
IF step was a collision:
    reward -= 1.0

ELSE:
    reward += 2 * (prev_dist - curr_dist)  # reward progress
    IF action is Rotate/Look:
        reward -= 0.05
    IF alternating LookUp <-> LookDown or RotateLeft <-> RotateRight:
        reward -= 0.2
    IF MoveAhead follows a LookUp/LookDown:
        reward += 0.1

IF goal reached (distance < 0.5m):
    reward += 10.0
    done = True
```

---

#### 💡 Why This Reward Structure?

- **Distance shaping** provides directional feedback at every step (progress = reward)
- **Collision penalty** discourages unsafe or aggressive movement
- **Look/spin penalties** reduce wasted or unnecessary actions
- **LookAhead bonus** encourages use of **depth perception** for navigation
- **Success reward** largest focus on goal-reaching behavior

This hybrid shaping strategy proved crucial in training learning-based agents effectively, especially in cluttered indoor scenes.

---

## Experiments

We implemented and evaluated a series of agents ranging from non-learning baselines to deep reinforcement learning models. All agents were trained and tested using the same episode structure, environment parameters and custom reward function.

---

### 🔹 Heuristic Agent

**Description:**  
A rule-based agent that tries to reduce the angle between the agent’s facing direction and the target goal position. If the angle is within a threshold, it moves forward.

**Decision Logic:**

```text
1. Compute angle to goal
2. IF large angle → rotate left/right
3. ELSE → MoveAhead
4. IF previous action failed → rotate right as recovery
```

**Strengths:**
- Capable of navigating in simple, obstacle-free scenes
- No training required

**Limitations:**
- Fails in presence of obstacles or may move into dead-ends
- Repetitive behavior (e.g., stuck loops or wall hugging)

---

### 🔹 Deep Q-Network (DQN)

**Description:**  
A model-free value-based reinforcement learning agent that learns a Q-function mapping states (e.g., RGB or depth frames) to expected cumulative rewards for each action.

**Architecture:**
- Input: Resized (128×128) visual frame (3/1/4 channels)
- CNN encoder → Fully connected layers → 5-dimensional Q-value output

**Training Details:**
- Epsilon-greedy exploration
- Experience replay buffer
- Target network updates for off-policy training
- Optimized via MSE loss between predicted and target Q-values

**Observations:**
- Struggles with sparse/delayed rewards common in PointNav
- Learns to move toward goals in most scenes with occasional instability in training (spikes in reward plot)
- Benefits significantly from well-shaped reward function but could also be impacted by gaps in our reward fn
- Not ideal for high-dimenstional Delayed rewards could hurt Q-value propagation
- Relies on experience replay which can break temporal correlations important in navigation

<img src="https://github.com/surya1701/RL-PointNav-project/raw/refs/heads/main/assets/dqn_arch.jpeg"/> 

---

### 🔹 Proximal Policy Optimization (PPO)

**Description:**  
An actor-critic reinforcement learning algorithm that optimizes a clipped surrogate objective using advantage-weighted updates. PPO is widely used in visually-rich and partially observable tasks due to its stability and sample efficiency.

**Architecture:**
- **Input:** Resized agent-view frame (RGB or RGB+Depth)
- **Encoder:** CNN for spatial feature extraction  
- **Memory Module:** LSTM added after CNN to preserve temporal context across steps  
- **Outputs:**
  - **Actor head:** Probability distribution over discrete actions  
  - **Critic head:** Value estimate for the current observation

**Why LSTM was Added:**  
PointNav tasks in RoboTHOR are **partially observable** — the agent doesn’t have access to a full map and must rely on previous frames to:
- Recall where it came from
- Avoid redundant actions like spinning in place
- Better understand how actions influence the environment over time

The **LSTM helps capture this temporal continuity**, giving the policy memory across steps.

**Training Details:**
- On-policy rollouts using **Generalized Advantage Estimation (GAE)**
- **Clipped PPO objective** to prevent unstable policy updates
- **Entropy regularization** encourages exploration
- LSTM hidden states are reset at the beginning of each episode and updated through each step

**Observations:**
- Reduces action loops and wandering behavior
- Increased computational overhead
- Needs proper hidden state handling
- Leads to smoother, more purposeful navigation policies

<img src="https://github.com/surya1701/RL-PointNav-project/raw/refs/heads/main/assets/ppo_arch.jpeg"/>

---

### Shared Implementation Infrastructure

All agents share a common training framework, including:

- Unified `RoboThorEnv` wrapper for consistent reset/step/state extraction
- Configurable CLI with agent type, scene selection, input mode
- Logging of episode rewards and video frames for visualization
- Modular code structure for reproducibility and extension

---

## Results

Performance is assessed via:

- **Average episode reward** (shaped via our custom function)
- **Qualitative behavior** through video analysis

---

### 📊 Reward Progression During Training

The following plot compares the training reward trajectories of our learning-based agents (DQN and PPO). PPO demonstrates more stable improvement, while DQN is more volatile but capable of sharp gains.

<img src="https://github.com/surya1701/RL-PointNav-project/raw/refs/heads/main/assets/dqn_plot.jpeg"/>
<img src="https://github.com/surya1701/RL-PointNav-project/raw/refs/heads/main/assets/ppo_plot.jpeg"/>

---

### 🎥 Episode Rollout Videos

The following videos show full PointNav episodes for each agent. These rollouts were recorded during training and demonstrate qualitative differences in decision-making and path quality.

| Table         | Couch       |
|-------------------------|------------------------|
|<video width="320" height="240" controls src="https://github.com/surya1701/RL-PointNav-project/raw/refs/heads/main/assets/dqn corner scene.mp4" type="video/mp4"></video> | <video width="320" height="240" controls  src="https://github.com/surya1701/RL-PointNav-project/raw/refs/heads/main/assets/DQN couch scene.mp4"></video> |


---

### 📝 Observations and Insights

- **Random Agent** often spins in place and makes no meaningful progress.
- **Heuristic Agent** performs better in obstacle-free layouts but fails in more complex rooms.
- **DQN Agent** demonstrates learning, but occasionally gets stuck or hesitates due to value approximation noise.
- **PPO Agent** is consistently better at exploiting the shaped reward and avoids repetitive loops.
- **Reward shaping** was critical to making training viable—especially the distance delta and spin penalties.

---

## References

1. [AI2-THOR](https://ai2thor.allenai.org)  
2. [AllenAct](https://github.com/allenai/allenact)  
3. [Target-Driven Navigation](https://arxiv.org/abs/1609.05143) — Zhu et al.  
4. [DQN - DeepMind](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
5. [PPO - Schulman et al.](https://arxiv.org/abs/1707.06347)  
6. [RoboTHOR Documentation](https://allenai.github.io/robothor)