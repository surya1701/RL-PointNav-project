# PointNav Agents in RoboTHOR

### Team Members
- Alice Smith  
- Bob Kumar  
- Charlie Nguyen  
- [Your Name Here]

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

Our project focuses on the **PointNav task** using the RoboTHOR environment—a collection of 3D indoor scenes developed for research on real-world navigation and transfer learning.

---

### RoboTHOR Environment and Dataset

We use the [AllenAct](https://github.com/allenai/allenact) implementation of the RoboTHOR PointNav dataset. Each episode specifies:

- A **scene** (e.g., `FloorPlan_Train5`)
- The agent's **initial position** and **orientation**
- A **goal position**
- Optionally, the **shortest path** between them for evaluation

#### 🔎 Example Episode (JSON)

```json
{
  "scene": "FloorPlan_Train5",
  "initial_position": {"x": 1.5, "y": 0.9, "z": -3.0},
  "initial_orientation": 90.0,
  "target_position": {"x": 2.25, "y": 0.9, "z": -1.75}
}
```

We process episodes into:
- **Single-scene** datasets (for overfitting/debugging)
- **Multi-scene** datasets (for generalization)

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

These parameters result in a **discrete action space**, where the agent can take one of the following actions at each step:

- `MoveAhead`
- `RotateLeft`
- `RotateRight`
- `LookUp`
- `LookDown`

---

### Observation Modalities

The agent receives **egocentric** observations in one of the following formats:
- **RGB only**
- **Depth only**
- **RGB + Depth** (concatenated as 4 channels)

#### 📷 Example Observations (placeholders)

| Top-Down Scene View     | Agent RGB View         | Agent Depth Image       |
|-------------------------|------------------------|-------------------------|
| ![](images/top_down.png) | ![](images/rgb_view.png) | ![](images/depth_view.png) |

These will be replaced with actual frames from rollout episodes for final deployment.

---

### Custom Reward Function

To guide the agent effectively during training, we implemented a custom **reward shaping** mechanism tailored for the PointNav task. Instead of using sparse rewards (e.g., +1 only on success), our design incorporates distance improvement, obstacle awareness, and behavior discouragement for inefficient or repetitive actions.

---

#### 📐 Reward Components

At each timestep, the reward is computed as:

\[
r = r_{\text{base}} + r_{\text{collision}} + r_{\text{distance}} + r_{\text{loop\_penalty}} + r_{\text{look\_penalty}} + r_{\text{depth\_bonus}} + r_{\text{success}}
\]

Where:

| Component            | Value / Condition                                                                 |
|----------------------|-----------------------------------------------------------------------------------|
| **Base step penalty**       | \( r_{\text{base}} = -0.01 \) per step                                      |
| **Collision penalty**       | \( r_{\text{collision}} = -1.0 \) if `lastActionSuccess = False`            |
| **Distance shaping**        | \( r_{\text{distance}} = 2 \cdot (\text{prev\_dist} - \text{curr\_dist}) \) |
| **Loop penalty**            | \( -0.2 \) if agent alternates between RotateLeft/Right or LookUp/Down     |
| **Look/Spin penalty**       | \( -0.05 \) if action is Look or Rotate                                     |
| **Depth navigation bonus**  | \( +0.1 \) if `MoveAhead` follows `LookUp` or `LookDown`                   |
| **Success reward**          | \( +20.0 \) if agent is within 0.5m of goal                                 |

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
    reward += 20.0
    done = True
```

---

#### 💡 Why This Reward Structure?

- **Dense shaping** provides directional feedback at every step (progress = reward)
- **Collision penalty** discourages unsafe or aggressive movement
- **Look/spin penalties** reduce wasted actions and dithering
- **LookAhead bonus** encourages use of **depth perception** for navigation
- **Success reward** reinforces goal-reaching behavior

This hybrid shaping strategy proved crucial in training learning-based agents effectively, especially in cluttered indoor scenes.

---

## Experiments

We implemented and evaluated a series of agents ranging from non-learning baselines to deep reinforcement learning models. All agents were trained and tested using the same episode structure, environment parameters, and custom reward function.

---

### 🔹 Random Agent

**Description:**  
The random agent selects one of the available discrete actions at each timestep with uniform probability.

**Purpose:**  
Acts as a naive baseline to quantify the difficulty of the task and assess the impact of having no navigation strategy.

**Behavior:**  
- Often collides with obstacles
- Spins in place
- Rarely reaches the goal

**Implementation Notes:**
- Stateless policy
- Uniform random over 5 actions: `MoveAhead`, `RotateLeft`, `RotateRight`, `LookUp`, `LookDown`

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
- Fails in presence of obstacles or dead-ends
- Repetitive behavior (e.g., stuck loops or wall hugging)

**Visual Aid:**  
📸 *[Placeholder for a sample top-down trajectory or agent view]*

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
- Target network updates
- Optimized via MSE loss between predicted and target Q-values

**Observations:**
- Learns to move toward goals in most scenes
- Occasional instability in training (spikes in reward plot)
- Benefits significantly from well-shaped reward function

**Visual Aid:**  
🧠 *[Placeholder: DQN architecture diagram]*  
🎥 *[Placeholder: DQN rollout video thumbnail]*

---

### 🔹 Proximal Policy Optimization (PPO)

**Description:**  
An actor-critic RL algorithm that updates policy and value networks based on clipped surrogate objectives and advantage estimation.

**Architecture:**
- Input: Resized frame (same as DQN)
- Shared CNN encoder → 
  - Actor head: probability distribution over actions  
  - Critic head: value estimate of current state

**Training Details:**
- On-policy updates with generalized advantage estimation (GAE)
- Clipped objective to limit large policy shifts
- Encourages exploration through entropy regularization

**Strengths:**
- More stable training compared to DQN
- Handles longer rollouts and dense rewards well

**Visual Aid:**  
📊 *[Placeholder: PPO vs DQN reward plot]*  
🎥 *[Placeholder: PPO video sample]*

---

### Shared Implementation Infrastructure

All agents share a common training framework, including:

- Unified `RoboThorEnv` wrapper for consistent reset/step/state extraction
- Configurable CLI with agent type, scene selection, input mode
- Logging of episode rewards and video frames for visualization
- Modular code structure for reproducibility and extension

---

Let me know when you're ready to move to the **Results** section next. I’ll include reward plots, side-by-side comparisons, videos, and conclusion-style observations.

---

## Results

We evaluate each agent on a held-out set of RoboTHOR PointNav episodes. Performance is assessed via:

- **Total episode reward** (shaped via our custom function)
- **Goal-reaching success** (within 0.5 meters)
- **Qualitative behavior** through video analysis

---

### 📊 Reward Progression During Training

The following plot compares the training reward trajectories of our learning-based agents (DQN and PPO). PPO demonstrates more stable improvement, while DQN is more volatile but capable of sharp gains.

![Training Reward Curves](images/reward_plot_comparison.png)

> *Figure: Smoothed episodic reward across 500 episodes for DQN and PPO. Reward range: -15 to 25.*

---

### 🧭 Top-Down Trajectory Visualization

Below are top-down visualizations of selected episodes, showing the start point (green), goal point (red), and agent trajectory over time. These visualizations provide insight into navigation quality and efficiency.

| Random Agent | Heuristic Agent | DQN Agent | PPO Agent |
|--------------|------------------|-----------|-----------|
| ![](images/random_topdown.png) | ![](images/heuristic_topdown.png) | ![](images/dqn_topdown.png) | ![](images/ppo_topdown.png) |

> *Figure: Sample trajectories in the same scene. Learning-based agents show more direct and successful paths to the goal.*

---

### 👁️ Egocentric View (Agent Perception)

Below are first-person observations from different agent rollouts, showing the RGB view used for decision-making. These illustrate how limited the input is—no map or external localization is available.

| DQN (RGB) | PPO (Depth) |
|-----------|-------------|
| ![](images/dqn_rgb_view.png) | ![](images/ppo_depth_view.png) |

> *Figure: Egocentric inputs processed by CNNs. The agent must learn to interpret spatial layout and depth cues visually.*

---

### 🎥 Episode Rollout Videos

The following videos show full PointNav episodes for each agent. These rollouts were recorded during evaluation and demonstrate qualitative differences in decision-making and path quality.

| Agent     | RGB Video | Depth Video |
|-----------|-----------|-------------|
| Random    | [▶️](videos/random_rgb.mp4) | [▶️](videos/random_depth.mp4) |
| Heuristic | [▶️](videos/heuristic_rgb.mp4) | [▶️](videos/heuristic_depth.mp4) |
| DQN       | [▶️](videos/dqn_rgb.mp4) | [▶️](videos/dqn_depth.mp4) |
| PPO       | [▶️](videos/ppo_rgb.mp4) | [▶️](videos/ppo_depth.mp4) |

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
