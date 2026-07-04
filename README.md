# MANHATON: Offline Reinforcement Learning Framework for Autonomous Data Cleaning

This repository contains the official implementation of the data preprocessing and state-trajectory repair environment detailed in **"MANHATON: An Offline Reinforcement Learning Framework for Autonomous Data Cleaning to Optimize State Transition Integrity in Clinical Datasets."**

[cite_start]The framework reformulates multi-variable missing data reconstruction in continuous clinical feature spaces as an offline Markov Decision Process (MDP)[cite: 1, 2, 7]. [cite_start]Policy convergence is driven by a localized, variance-weighted $L_1$ Manhattan Distance penalty structure designed to maintain natural population distributions and multi-variable covariance boundaries without introducing out-of-distribution structural noise[cite: 1, 2, 9].

## Repository Architecture

```text
├── data/
│   ├── heart.csv               # Original raw clinical dataset
│   └── heart_rl_cleaned.csv    # Final high-fidelity imputed matrix
├── src/
│   └── manhaton_cleaner.py     # State step execution loop and reward environment
├── README.md                   # Repository documentation
└── requirements.txt            # Software dependency manifest
