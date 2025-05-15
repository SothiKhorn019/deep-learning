#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import npfl138

npfl138.require_version("2425.11")

MODEL_PATH = "reinforce_pixels.pt"

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Running in ReCodEx"
)
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument(
    "--hidden_layer_size", default=128, type=int, help="Size of hidden layer."
)
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument(
    "--gamma", default=0.99, type=float, help="Discount factor for returns."
)


class Agent:
    # Use accelerator if available
    device = npfl138.trainable_module.get_auto_device()

    def __init__(
        self, env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace
    ) -> None:
        # Observation shape: (height, width, channels)
        obs_shape = env.observation_space.shape
        h, w, in_channels = obs_shape
        n_actions = env.action_space.n

        # Build a small convolutional policy network
        # conv layers: (in_channels)->16->32->32, flatten, FC->hidden->ReLU->actions
        # Compute conv output size by running a dummy tensor
        conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            conv_out_size = conv_layers(dummy).shape[1]

        # Policy network
        self._policy = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(conv_out_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, n_actions),
        ).to(self.device)

        # Baseline network: same conv backbone, separate head to single value
        self._baseline = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(conv_out_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
        ).to(self.device)

        # Optimizers
        self._optimizer_policy = torch.optim.Adam(
            self._policy.parameters(), lr=args.learning_rate
        )
        self._optimizer_value = torch.optim.Adam(
            self._baseline.parameters(), lr=args.learning_rate
        )
        self._value_loss = torch.nn.MSELoss()

    @npfl138.rl_utils.typed_torch_function(
        device, torch.float32, torch.int64, torch.float32
    )
    def train(
        self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor
    ) -> None:
        # States shape: (batch, H, W, C) -> convert to (batch, C, H, W) and normalize
        x = states.permute(0, 3, 1, 2) / 255.0

        # Compute baseline values (no grad for advantage)
        values = self._baseline(x).squeeze(1)
        advantages = returns - values.detach()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy forward
        logits = self._policy(x)
        log_probs = F.log_softmax(logits, dim=1)
        selected = log_probs[torch.arange(len(actions)), actions]
        policy_loss = -torch.mean(selected * advantages)

        # Update policy network
        self._optimizer_policy.zero_grad()
        policy_loss.backward()
        self._optimizer_policy.step()

        # Update baseline network: minimize MSE to returns
        value_loss = self._value_loss(values, returns)
        self._optimizer_value.zero_grad()
        value_loss.backward()
        self._optimizer_value.step()

    @npfl138.rl_utils.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        # Greedy policy: get action probabilities
        x = states.permute(0, 3, 1, 2) / 255.0
        logits = self._policy(x)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()


def main(env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        agent = Agent(env, args)
        # Load pretrained
        checkpoint = torch.load(MODEL_PATH, map_location=Agent.device)
        agent._policy.load_state_dict(checkpoint["policy"])
        agent._baseline.load_state_dict(checkpoint["baseline"])
        agent._policy.eval()
        agent._baseline.eval()

        # Infinite evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # Greedy action
                probs = agent.predict(np.array([state], dtype=np.float32))[0]
                action = int(np.argmax(probs))
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        return

    # Training mode
    agent = Agent(env, args)
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Run one episode
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                probs = agent.predict(np.array([state], dtype=np.float32))[0]
                action = int(np.random.choice(len(probs), p=probs))
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

            # Compute returns
            G = 0.0
            returns = []
            for r in reversed(rewards):
                G = r + args.gamma * G
                returns.insert(0, G)

            # Accumulate batch
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        # Train on batch
        agent.train(
            np.array(batch_states, dtype=np.float32),
            np.array(batch_actions, dtype=np.int64),
            np.array(batch_returns, dtype=np.float32),
        )

    # Save the trained model for ReCodEx evaluation
    torch.save(
        {
            "policy": agent._policy.state_dict(),
            "baseline": agent._baseline.state_dict(),
        },
        MODEL_PATH,
    )

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            probs = agent.predict(np.array([state], dtype=np.float32))[0]
            action = int(np.argmax(probs))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl138.rl_utils.EvaluationEnv(
        gym.make("npfl138/CartPolePixels-v1"), main_args.seed, main_args.render_each
    )

    main(main_env, main_args)
