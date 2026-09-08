# Channel Estimation Using Deep Learning

A flexible, deep learning-based **baseline framework** for wireless channel estimation. This repository provides a simple, modular architecture tailored for both **pilot-based** and **pilot-free** configurations, allowing you to easily integrate, modify, and benchmark different wireless communication systems.

##  Overview

Accurate channel estimation is critical for modern wireless systems. This project introduces a unified simulation environment that decouples the core neural network architectures from specific wireless configurations. 

It serves as a foundational baseline where you can implement alternative wireless systems or modify the pipeline for highly complex channel estimation scenarios.

### Key Features
* **Modular Baseline:** Simple architecture designed to be easily modified or extended for custom wireless setups.
* **Pilot & Pilot-Free Support:** Ready-to-use models for standard pilot-based estimation as well as pilot-free deep learning configurations.
* **DL Architectures:** Implemented using Deep Neural Networks, Fully Connected CNNs (FC-CNN), and Autoencoders (AE).
* **MATLAB Powered:** Built entirely as high-performance MATLAB simulation scripts.

---

##  Implemented Wireless Systems & Models

The repository currently includes full implementations for the following environments:
* **MIMO-OFDM System:** Channel estimation optimized for Multiple-Input Multiple-Output configurations using Cyclic Prefix OFDM.
* **UAV CP-OFDM System:** Channel estimation tailored for Unmanned Aerial Vehicle (UAV) communication profiles using FC-CNN.
* **Pilot-Free Autoencoder Model:** An unsupervised/self-supervised learning approach using Autoencoders for pilot-free estimation.

---

##  File Structure

* `Pilot Free Channel Estimation Using DL (Architecture).m` — The core deep learning backbone and neural network layers.
* `Pilot Free Channel Estimation Using DL.m` — Main execution pipeline for pilot-free deep learning estimation.
* `Pilot Free Channel Estimation Using Auto Encoder.m` — Implementation utilizing an Autoencoder architecture.
* `Channel Estimation of MIMO System (CPOFDM).m` — Simulation environment for MIMO CP-OFDM systems.
* `UAV CP-OFDM Using FC CNN.m` — Dedicated model for UAV wireless communication using Fully Connected CNNs.

---

##  How to Use

1. Open **MATLAB** and navigate to the cloned directory.
2. To run the base architecture, execute:
   ```matlab
   run('Pilot Free Channel Estimation Using DL.m')
   ```
3. **To Implement Your Own System:** 
   Open `Pilot Free Channel Estimation Using DL (Architecture).m` to view the base deep learning layout. You can feed your own generated channel matrices (H) or simulated data into this framework to test new communication environments (e.g., 6G RIS, mmWave).

---

##  Contributing

Contributions are welcome! If you implement a new wireless system or adapt these models for a unique channel environment, feel free to open a Pull Request or share your findings.
