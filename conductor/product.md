# Product Guide: Continuon AI

## Vision
To build a platform that decouples general robot intelligence from hardware, enabling continuous learning with multiple mediums or without training. The system leverages the ability to "imagine" via symbolic search and utilizes State Space Models (SSMs) for each level of the HOPE Architecture (Fast, Mid, and Slow loops). The slow loop is trained in the cloud using ContinuonAI-generated RLDS datasets and auto-annotated YouTube data, with the ultimate mission of helping humanity.

## Target Audience
- **Robot Owners/Operators:** Individuals and small teams seeking "smart" personal robots that can learn and adapt to their specific needs.
- **Robotics Researchers & Developers:** Open-source communities and hardware partners looking for a standardized, morphology-agnostic runtime and data loop for advanced AI research.

## Core Features & Priorities
1.  **Multi-Scale HOPE Loops:** Functional implementation of Fast (reflex), Mid (tactical), and Cloud-based Slow (strategic) SSM loops, optimized specifically for the Raspberry Pi 5 (8GB) memory and thermal constraints.
    - **Sequential Seed Training:** Capability to train the "seed" model directly on the Pi 5 (8GB) using a sequential component loading strategy to manage memory.
2.  **Unified RLDS Data Pipeline:** Standardized ingestion and processing of teleoperation data (XR), on-device sensor streams, and external video datasets for robust cloud-based training.
3.  **On-Device Symbolic Search:** Implementation of the "Searcher" module on the edge (Pi 5) to enable the robot to perform "System 2" thinking—simulating and inventing solutions locally without explicit prior training for every scenario.
4.  **Unified Discovery & Role-Based Orchestration:** Seamless robot discovery across CLI and mobile apps with integrated Role-Based Access Control (RBAC). The Companion app automatically adapts its interface (Creator vs. Consumer) to expose training, model management, and system context tools.
5.  **Deterministic Safety Kernel (Ring 0):** A dedicated, high-priority safety process that enforces physical laws and safety rules independently of the primary AI brain.
6.  **Brain Studio Cognition Dashboard:** Real-time visualization of the robot's internal reasoning, HOPE loop states, and an autonomous skill-teaching curriculum.

## Synchronization & Cloud Integration
- **Operational Autonomy:** The robot operates fully autonomously on the edge.
- **Data Ingestion:** Users maintain control by manually triggering the upload of RLDS episodes to the cloud.
- **Lifecycle Management:** Model updates and system improvements are automatically pulled and installed when the device is docked or charging.

## Hardware Support (Shells)
- **Primary Development Platform:** Raspberry Pi 5-based robotic arms (e.g., SO-ARM100/101).
- **Extended Compatibility:** Architected to support Mobile Bases/Rovers (e.g., Donkey Car) and commercial platforms (e.g., Hello Robot Stretch, Unitree Go2) via a standardized Hardware Abstraction Layer (HAL).
