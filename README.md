# DarkBotX - Tactile and Proprioceptive Shape Modeling for Planning Visionless Grasp with Reinforcement Learning in a Soft Robotic Manipulator

This repository contains the code and resources for **DarkBotX**, an AI-driven, visionless soft robotic manipulator developed for the **MTE 4200 Project and Thesis** at the **Department of Mechatronics Engineering, Rajshahi University of Engineering & Technology (RUET)**. DarkBotX operates in highly occluded environments where traditional optical sensors fail by leveraging haptic data, a custom Radial Boundary Reconstruction (RBR) algorithm, and Deep Reinforcement Learning (RL) to compute and execute stable grasps.

## Project Overview

The **DarkBotX** project addresses the limitations of vision-dependent robotic manipulation by eliminating dependency on costly and computationally heavy optical sensors (cameras/LiDAR). Instead, the system relies entirely on edge-computed haptic exploration using Force Sensitive Resistors (FSRs) and proprioceptive feedback to estimate 3D geometries and plan optimal grasps.

Key components and modules include:
- **Soft Robotic Manipulator** featuring compliant silicone fingers and variable stiffness design
- **Force Sensitive Resistors (FSRs)** and **ADS1115 ADC** for high-precision haptic data acquisition
- **Raspberry Pi 5** for low-latency edge computing and sensor processing
- **Genesis Sim** for high-fidelity, GPU-accelerated physics simulation and reinforcement learning training

## Features

- **Visionless Grasping:** Successfully plans and executes grasps in completely occluded or vision-denied spaces.
- **Genesis Sim & RSL-RL Integration:** Implements a specialized parallelized `GraspEnv` environment using `rsl_rl` for Actor-Critic PPO training.
- **Advanced Signal Filtering:** Utilizes a digital Butterworth filter achieving high Signal-to-Noise Ratios (SNR) on haptic lines.
- **Low-Latency Edge Execution:** Runs inference and control loops on a Raspberry Pi 5 under strict timing constraints (<3.74 ms latency).

## Hardware Components

1. **Raspberry Pi 5** - Edge computing platform handling low-level control loops and sensor reading.
2. **ADS1115 ADC** - High-resolution analog-to-digital converter for FSR signals.
3. **MG996R Servos / Actuators** - Driving joint and finger mechanics.
4. **Force Sensitive Resistors (FSRs)** - Distributed tactile elements embedded within the soft manipulator structure.

## Repository Structure

- `simulation/` - Contains core simulation environments (`environment.py`) built on Genesis Sim.
- `hardware/` - URDF description files, mechanical CAD models, and mold STL assets.
- `rl_training/` - PPO reward configurations, policy networks, and training logs.
- `edge_control/` - Low-latency Python deployment scripts designed for the Raspberry Pi hardware.

## Installation and Setup

1. **Clone this repository** to your local machine:
   ```bash
   git clone [https://github.com/abrar-faiyaz-anan/darkbotx.git]
   cd darkbotx
   pip install -r requirements.txt
   cd simulation
   python environment.py

## How It Works

1. The **Genesis Sim** environment initializes rigid and soft-body physics, spawning target test objects with randomized orientations and positions.
2. The actor-critic network processes spatial and tactile states to compute optimal wrist movements and gripping forces.
3. On physical deployment, the **Raspberry Pi 5** reads raw analog values from the FSRs via the ADS1115, applies filtering, and passes them to the control architecture.

## Usage

- Run simulation scripts locally to evaluate policy performance and test reward functions.
- Deploy edge controller scripts to interface directly with the physical soft manipulator setup.

## Future Scope & Impact

This project highlights a sustainable hardware-software co-design by replacing heavy 250W+ GPU vision arrays with a low-power 15W edge-compute topology. Future expansions include:
- **High-Resolution Tactile Arrays** for enhanced boundary mapping.
- **Advanced Sim-to-Real Domain Adaptation** to narrow the physical deployment gap.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
