#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import npfl138
npfl138.require_version("2425.11")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=3, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor for returns.")


class Agent:
    # Use an accelerator if available.
    device = npfl138.trainable_module.get_auto_device()

    def __init__(self, env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model of the policy. Note that the shape
        # of the observations is available in `env.observation_space.shape`
        # and the number of actions in `env.action_space.n`.
        #
        # Apart from the policy network defined in `reinforce` assignment, you
        # also need a value network for computing the baseline, returning
        # a single output with no activation.
        #
        # Using Adam optimizer with the given `args.learning_rate` for both models
        # is a good default.
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        # Policy network
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, n_actions)
        ).to(self.device)
        # Baseline network
        self._baseline = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)
        # Optimizers
        self._optimizer_policy = torch.optim.Adam(self._policy.parameters(), lr=args.learning_rate)
        self._optimizer_value = torch.optim.Adam(self._baseline.parameters(), lr=args.learning_rate)
        # Loss for baseline
        self._value_loss = torch.nn.MSELoss()

    # The `npfl138.rl_utils.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl138.rl_utils.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform training.
        # You should:
        # - compute the predicted baseline using the baseline model,
        # - train the policy model, using `returns` - `predicted_baseline` as
        #   advantage estimate,
        # - train the baseline model to predict `returns`.
        #
        # Note that predicting returns in 0-500 range is challenging for the network, given
        # that the default initialization tries to keep variance -- it might be helpful for
        # the network if you predict returns in a smaller range.
       
        # Compute baseline values
        values = self._baseline(states).squeeze(1)
        # Compute advantages
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Policy loss: negative log likelihood weighted by advantage
        logits = self._policy(states)
        
        log_probs = F.log_softmax(logits, dim=1)
        selected = log_probs[torch.arange(len(actions)), actions]
        policy_loss = -torch.mean(selected * advantages)
        # Update policy
        self._optimizer_policy.zero_grad()
        policy_loss.backward()
        self._optimizer_policy.step()
        # Value loss: MSE between returns and predicted baseline
        value_loss = self._value_loss(values, returns)
        # Update baseline
        self._optimizer_value.zero_grad()
        value_loss.backward()
        self._optimizer_value.step()

    @npfl138.rl_utils.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        # TODO(reinforce): Define the prediction method returning policy probabilities.
        
        logits = self._policy(states)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()


def main(env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Construct the agent.
    agent = Agent(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform an episode.
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                # TODO(reinforce): Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                probs = agent.predict(np.array([state], dtype=np.float32))[0]
                action = int(np.random.choice(len(probs), p=probs))

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute returns by summing rewards.
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + args.gamma * G
                returns.insert(0, G)

            # TODO(reinforce): Append states, actions and returns to the training batch.
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        # TODO(reinforce): Train using the generated batch.
        agent.train(
            np.array(batch_states, dtype=np.float32),
            np.array(batch_actions, dtype=np.int64),
            np.array(batch_returns, dtype=np.float32)
        )

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(reinforce): Choose a greedy action.
            probs = agent.predict(np.array([state], dtype=np.float32))[0]
            action = int(np.argmax(probs))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl138.rl_utils.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
