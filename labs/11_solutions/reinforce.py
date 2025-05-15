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
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
parser.add_argument("--hidden_layers", default=[128], nargs='+', type=int, help="Hidden layer sizes.")
parser.add_argument("--activation", default="ReLU", type=str, help="Activation function (e.g. ReLU, Tanh)")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--entropy_coef", default=0.01, type=float, help="Entropy regularization coefficient.")

class Agent:
    # Use an accelerator if available.
    device = npfl138.trainable_module.get_auto_device()

    def __init__(self, env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model of the policy. Note that the shape
        # of the observations is available in `env.observation_space.shape`
        # and the number of actions in `env.action_space.n`.
        obs_dim = env.observation_space.shape[0]
        action_n = env.action_space.n
        
        self._entropy_coef = args.entropy_coef
        
        layers = []
        input_dim = obs_dim
        activation_cls = getattr(torch.nn, args.activation)
        for size in args.hidden_layers:
            layers.append(torch.nn.Linear(input_dim, size))
            layers.append(activation_cls())
            if args.dropout > 0:
                layers.append(torch.nn.Dropout(args.dropout))
            input_dim = size
        layers.append(torch.nn.Linear(input_dim, action_n))
        self._policy = torch.nn.Sequential(*layers).to(self.device)

        # TODO: Define an optimizer. Using `torch.optim.Adam` optimizer with
        # the given `args.learning_rate` is a good default.
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=args.learning_rate)

        # TODO: Define the loss (most likely some `torch.nn.*Loss`).
        self._loss = torch.nn.CrossEntropyLoss(reduction="none")

    # The `npfl138.rl_utils.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl138.rl_utils.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform training, using the loss from the REINFORCE algorithm.
        # The easiest approach is to construct the cross-entropy loss with
        # `reduction="none"` argument and then weight the losses of the individual
        # examples by the corresponding returns.
        
        # Forward pass
        logits = self._policy(states)
        log_probs = F.log_softmax(logits, dim=1)
        # Negative log-probabilities for taken actions
        neg_logp = -torch.gather(log_probs, dim=1, index=actions.unsqueeze(1)).squeeze(1)
        # Policy loss weighted by returns
        policy_loss = (neg_logp * returns).mean()
        # Entropy bonus
        probs = torch.exp(log_probs)
        entropy = -(log_probs * probs).sum(dim=1).mean()
        # Use stored entropy coefficient
        loss = policy_loss - self._entropy_coef * entropy

        # Backpropagation with gradient clipping
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), max_norm=1.0)
        self._optimizer.step()

    @npfl138.rl_utils.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Define the prediction method returning policy probabilities.
        
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
                # TODO: Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                probs = agent.predict(np.array(state, dtype=np.float32)[None, :])[0]
                action = np.random.choice(len(probs), p=probs)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO: Compute returns by summing rewards.
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = G + r
                returns.insert(0, G)

            # TODO: Append states, actions and returns to the training batch.
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        # TODO: Train using the generated batch.
        states_arr = np.array(batch_states, dtype=np.float32)
        actions_arr = np.array(batch_actions, dtype=np.int64)
        returns_arr = np.array(batch_returns, dtype=np.float32)
        returns_arr = (returns_arr - returns_arr.mean()) / (returns_arr.std() + 1e-8)

        agent.train(states_arr, actions_arr, returns_arr)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action.
            probs = agent.predict(np.array(state, dtype=np.float32)[None, :])[0]
            action = int(np.argmax(probs))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl138.rl_utils.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
