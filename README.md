# Simbot-Navigation-Agent: A Dialog-Enabled VLN Framework

An implementation of a dialog-enabled agent for visual-language navigation (VLN) in a simulated 3D environment. The agent learns to follow natural language instructions and can ask clarifying questions when faced with ambiguity.

---

## 1. Project Overview

This project builds an embodied AI agent that can navigate a realistic 3D home environment based on natural language instructions like "Go to the microwave in the kitchen." The core challenge is to ground language commands in visual perception and a sequence of actions. A key feature of this project is making the agent **dialog-enabled**, allowing it to seek clarification when instructions are ambiguous, a crucial part of the Simbot challenge.

---

## 2. Core Objectives

-   To set up and interact with a simulated 3D environment.
-   To implement a cross-modal agent that connects vision (what it sees) and language (the instructions).
-   To train a navigation policy using imitation learning on a standard VLN dataset.
-   To implement a simple dialog manager that allows the agent to ask for help when it's uncertain.

---

## 3. Methodology

#### Phase 1: Environment and Dataset

1.  **Simulator**: We will use **AI2-THOR**, a popular and powerful open-source simulator that provides realistic indoor environments with interactable objects. It's a great choice for this type of embodied AI task.
2.  **Dataset**: We'll use the **Room-to-Room (R2R)** dataset, which is the standard benchmark for VLN. It consists of navigation instructions paired with ground-truth paths through the environment, making it perfect for training our agent.

#### Phase 2: Agent Architecture

The agent will be built with a modular, cross-modal architecture.

1.  **Vision Module**: At each step, the agent receives a first-person image of the environment. We will use a pre-trained **ResNet-50** to extract a compact feature vector from this image.
2.  **Language Module**: The navigation instruction (e.g., "Walk past the table and stop at the fridge") will be encoded into a vector using a pre-trained **BERT** or a simpler **GRU/LSTM** network.
3.  **Policy Module (The "Brain")**:
    -   This module decides which action to take at each step (`move_forward`, `turn_left`, `turn_right`, `stop`).
    -   We will implement a recurrent model (e.g., an LSTM) that takes the current visual features and the instruction embedding as input.
    -   It will be trained via **imitation learning** (or behavioral cloning), where the model learns to mimic the expert actions provided in the R2R dataset.

#### Phase 3: Dialog Component

This is what makes the agent "dialog-enabled."

1.  **Ambiguity Detection**: We'll implement a simple heuristic. If the navigation policy outputs low-confidence scores for all possible actions for several consecutive steps (e.g., it's "hesitating" between turning left or right at a fork), the agent will consider itself "stuck" or "uncertain."
2.  **Clarification Question**: When uncertainty is detected, the agent will pause its navigation and output a pre-defined clarification question, such as "I'm not sure where to go. Should I turn left or right?".

#### Phase 4: Evaluation

We will evaluate our agent using standard VLN metrics on an unseen set of navigation paths.

-   **Success Rate (SR)**: The percentage of paths where the agent stops within a certain distance of the target.
-   **Path Length (PL)**: The total length of the path taken by the agent.
-   **Success weighted by Path Length (SPL)**: A metric that balances success and efficiency, penalizing the agent for taking unnecessarily long routes.

---

## 4. Project Structure
```text
/simbot-navigation-agent
|
|-- /agent/
|   |-- vision_module.py      # ResNet-based feature extractor
|   |-- language_module.py    # Instruction encoder
|   |-- policy_module.py      # The recurrent navigation policy model
|   |-- dialog_manager.py     # Logic for asking clarification questions
|
|-- /environment/
|   |-- setup_ai2thor.py      # Script to initialize and interact with the simulator
|
|-- train.py                  # Main script for training the agent with imitation learning
|-- evaluate.py               # Script to run evaluation and report metrics
|
|-- requirements.txt
|-- README.md
```
