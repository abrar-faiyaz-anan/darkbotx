# darkbotx
Tactile and Proprioceptive Shape Modeling for Planning Visionless Grasp with Reinforcement Learning in a Soft Robotic Manipulator


DarkBotX is an visionless soft robotic manipulator designed to operate in highly occluded environments where traditional optical sensors fail. By leveraging Force Sensitive Resistors (FSRs), a custom Radial Boundary Reconstruction (RBR) algorithm and Reinforcement Learning (RL), the manipulator "feels" its surroundings to successfully compute and execute stable grasps.

## 🚀 Key Performance Metrics
* **Physical Grasp Success Rate:** 87.6% across varied test geometries (cubes, cylinders).
* **Inference Latency:** < 3.74 ms (Running locally on a Raspberry Pi 5).
* **Sensor Noise Reduction:** 46.2 dB SNR achieved via a custom digital Butterworth filter.

## 🧠 System Architecture

### 1. Hardware & Edge Computing
* **Microcontroller:** Raspberry Pi 5
* **Actuation:** MG996R Servos controlled via PWM/I2C
* **Sensory Input:** Force Sensitive Resistors (FSRs) read through an ADS1115 ADC
* **End Effector:** Custom 3D printed and casted soft tendon driven fingers

### 2. Software & Control Stack
* **Simulation Environment:** [Genesis Sim](https://github.com/Genesis-Embodied-AI/Genesis) for highly accurate soft body physics modeling.
* **Algorithm:** Custom Radial Boundary Reconstruction (RBR) for point-cloud generation from haptic feedback.
* **Reinforcement Learning:** Trained using `RSL_RL` (Actor Critic formulation) in PyTorch.
* **Geometry Processing:** `mapbox-earcut` and `trimesh` used for spatial triangulation.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abrar-faiyaz-anan/darkbotx.git](https://github.com/abrar-faiyaz-anan/darkbotx.git)
   cd darkbotx
