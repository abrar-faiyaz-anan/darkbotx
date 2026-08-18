# darkbotx
**Tactile and Proprioceptive Shape Modeling for Planning Visionless Grasp with Reinforcement Learning in a Soft Robotic Manipulator**

DarkBotX is a visionless robotic manipulator designed to compute and execute stable grasps without traditional optical sensors. By leveraging local periphery tracking and Reinforcement Learning (RL), the manipulator aligns itself and securely grasps target geometries.

## 🚀 Key Features & Performance
* **Custom RL Environment:** Implements a specialized `GraspEnv` inheriting from `rsl_rl` for highly parallelized training.
* **Intelligent Reward Structure:** Utilizes a multi-stage reward function that optimizes for geometric alignment and target width projection prior to execution.
* **Physics Simulation:** Built on Genesis Sim for accurate, GPU-accelerated rigid-body physics.

## 🧠 System Architecture

### 1. Software & Training Stack
* **Simulation Environment:** [Genesis Sim](https://github.com/Genesis-Embodied-AI/Genesis).
* **Reinforcement Learning:** Trained using Proximal Policy Optimization (PPO) via `rsl_rl`.
* **Geometry Processing:** `trimesh` used for URDF mesh projection, 2D boundary resampling, and spatial triangulation.

### 2. Hardware & Edge Computing
* **Controller:** Raspberry Pi / Edge Compute Device
* **Actuation:** Servos controlled via PWM/I2C
* **Sensory Input:** Haptic data processed via I2C ADC

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abrar-faiyaz-anan/darkbotx.git](https://github.com/abrar-faiyaz-anan/darkbotx.git)
   cd darkbotx

   pip install -r requirements.txt

   cd simulation
python environment.py
