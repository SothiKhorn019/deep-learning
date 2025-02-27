#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument(
    "--data_path",
    default="numpy_entropy_data.txt",
    type=str,
    help="Data distribution path.",
)
parser.add_argument(
    "--model_path",
    default="numpy_entropy_model.txt",
    type=str,
    help="Model distribution path.",
)
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Evaluation in ReCodEx."
)
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    data_list = []
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            data_list.append(line)

    # Count occurrences and compute probabilities
    data_counts = {}
    for item in data_list:
        if item in data_counts:
            data_counts[item] += 1
        else:
            data_counts[item] = 1

    total_data = sum(data_counts.values())
    data_distribution = {key: count / total_data for key, count in data_counts.items()}

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    data_probs = np.array(list(data_distribution.values()))

    # TODO: Load model distribution, each line `string \t probability`.
    model_distribution = {}
    with open(args.model_path, "r") as model:
        for line in model:
            # line = line.rstrip("\n")
            # TODO: Process the line, aggregating using Python data structures.
            key, prob = line.strip().split("\t")
            model_distribution[key] = float(prob)

    # TODO: Create a NumPy array containing the model distribution.
    model_probs_list = []
    for key in data_distribution:
        if key in model_distribution:
            model_probs_list.append(model_distribution[key])
        else:
            model_probs_list.append(0)

    model_probs = np.array(model_probs_list)
    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(data_probs * np.log(data_probs))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.

    # If any model probability is missing
    if np.any(model_probs == 0):
        crossentropy = np.inf
        kl_divergence = np.inf
    else:
        crossentropy = -np.sum(data_probs * np.log(model_probs))
        kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(main_args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
