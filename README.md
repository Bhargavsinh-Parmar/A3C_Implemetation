Atari AI A3C Agent


 Atari AI A3C Agent

This repository contains an implementation of an Atari AI Agent using Asynchronous Advantage Actor-Critic (A3C) with PyTorch and Gymnasium. The agent is designed to learn and play Atari games, specifically the "KungFuMasterDeterministic-v0" environment. It uses convolutional neural networks (CNNs) to process game frames and reinforcement learning techniques to train the agent.

 Requirements

Make sure to install the following dependencies before running the code:

 Python Packages:
- gymnasium: For building and simulating the Atari game environment.
- ale-py: Atari Learning Environment (ALE) for interacting with Atari games.
- OpenCV: For image preprocessing (resizing, color conversion).
- numpy: For numerical operations.
- torch: PyTorch framework for building and training the neural network.
- tqdm: For progress tracking during training.

 Installation Commands:
```bash
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
pip install ale-py
apt-get install -y swig
pip install gymnasium[box2d]
pip install opencv-python-headless
pip install tqdm
```

 Structure

 1. Neural Network Architecture

The core of the AI is the neural network defined in the `Network` class. It consists of:
- 3 convolutional layers: For feature extraction from game frames.
- 2 fully connected layers: For processing features and generating action probabilities and state values.

 2. Preprocessing Environment

The `PreprocessAtari` class preprocesses the raw game frames:
- Resizes the frames to 42x42 pixels.
- Converts them to grayscale for simplicity (optional).
- Stacks the last 4 frames to capture temporal information.

 3. A3C Agent

The `Agent` class implements the A3C algorithm:
- It uses the neural network to predict action probabilities and state values.
- Policy gradient loss and value loss are combined to train the network.
- The agent is capable of interacting with multiple environments at the same time using parallel processing.

 4. Training and Evaluation

- The agent is trained using 10 parallel environments to increase learning speed.
- Training runs for 3000 iterations, with evaluation every 1000 iterations.
- After training, the agent can be evaluated on a single episode, and results are visualized.

 5. Visualization

The `show_video_of_model` function records and displays a video of the agent playing the game after training.

 How to Run

1. Set up the environment: Install dependencies as shown above.
2. Run the training code: Train the A3C agent on the Atari "KungFuMasterDeterministic-v0" environment.
3. Visualize the results: After training, use the `show_video_of_model` function to generate a video of the agent playing the game.

 Example

```bash
 Train the agent for 3000 iterations
python train_agent.py

 Show the video of the agent playing the game
python render_video.py
```

 Hyperparameters

The default hyperparameters for training are:
- `learning_rate = 1e-4`: Controls the step size during optimization.
- `discount_factor = 0.99`: Discount rate for future rewards in the reinforcement learning algorithm.
- `number_environments = 10`: Number of environments running in parallel for faster training.

 References

This project is inspired by the A3C algorithm, and leverages the power of Gymnasium and PyTorch to build a scalable reinforcement learning model capable of solving Atari games.

 License

This project is open-source and available under the [MIT License](LICENSE).
