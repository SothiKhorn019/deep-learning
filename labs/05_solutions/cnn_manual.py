#!/usr/bin/env python3
import argparse

import numpy as np
import torch

import npfl138
npfl138.require_version("2425.5")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="5-3-2,10-3-2", type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the implementation.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Convolution:
    def __init__(
        self, filters: int, kernel_size: int, stride: int, input_shape: list[int], verify: bool,
    ) -> None:
        # Create a convolutional layer with the given arguments and given input shape.
        # Note that we use NHWC format, so the MNIST images have shape [28, 28, 1].
        self._filters = filters
        self._kernel_size = kernel_size
        self._stride = stride
        self._verify = verify

        # Here the kernel and bias variables are created, the kernel has shape
        # [kernel_height, kernel_width, input_channels, output_channels], bias [output_channels].
        self._kernel = torch.nn.Parameter(torch.randn(kernel_size, kernel_size, input_shape[2], filters) * 0.1)
        self._bias = torch.nn.Parameter(torch.zeros(filters))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO: Compute the forward propagation through the convolution
        # with ReLU activation, and return the result.
        #
        # In order for the computation to be reasonably fast, you cannot
        # manually iterate through the individual pixels, batch examples,
        # input filters, or output filters. However, you can manually
        # iterate through the kernel size.
        # inputs shape: [N, H, W, C_in]
        
        N, H, W, C_in = inputs.shape
        K = self._kernel_size
        s = self._stride
        # Compute output dimensions for valid padding.
        H_out = (H - K) // s + 1
        W_out = (W - K) // s + 1

        # Initialize output (before activation).
        conv = torch.zeros((N, H_out, W_out, self._filters), dtype=inputs.dtype, device=inputs.device)
        # For each kernel element, accumulate contribution from the corresponding input patch.
        for u in range(K):
            for v in range(K):
                # Extract patch: shape [N, H_out, W_out, C_in]
                patch = inputs[:, u:u + s * H_out:s, v:v + s * W_out:s, :]
                # Multiply the patch with the kernel slice [C_in, filters] and sum over C_in.
                conv += torch.einsum('nhwc,co->nhwo', patch, self._kernel[u, v, :, :])
        # Add bias and apply ReLU.
        conv = conv + self._bias 
        output = torch.relu(conv)

        # If requested, verify that output contains a correct value.
        if self._verify:
            reference = torch.relu(torch.nn.functional.conv2d(
                inputs.movedim(-1, 1), self._kernel.permute(3, 2, 0, 1), self._bias, self._stride)).movedim(1, -1)
            np.testing.assert_allclose(output.detach().numpy(), reference.detach().numpy(), atol=1e-4,
                                       err_msg="Forward pass differs!")

        return output

    def backward(
        self, inputs: torch.Tensor, outputs: torch.Tensor, outputs_gradient: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        # TODO: Given this layer's inputs, this layer's outputs,
        # and the gradient with respect to the layer's outputs,
        # compute the derivatives of the loss with respect to
        # - the `inputs` layer,
        # - `self._kernel`,
        # - `self._bias`.
        
        N, H, W, C_in = inputs.shape
        K = self._kernel_size
        s = self._stride
        H_out = (H - K) // s + 1
        W_out = (W - K) // s + 1

        # Because forward applies ReLU, multiply by its derivative: 1 if output > 0, else 0.
        delta = outputs_gradient * (outputs > 0).to(outputs_gradient.dtype)
        
        # Compute bias gradient: sum over batch and spatial dimensions.
        bias_gradient = delta.sum(dim=(0, 1, 2)) 
        
        # Compute kernel gradient.
        kernel_gradient = torch.zeros_like(self._kernel)
        for u in range(K):
            for v in range(K):
                # Extract the corresponding input patch: [N, H_out, W_out, C_in]
                patch = inputs[:, u:u + s * H_out:s, v:v + s * W_out:s, :]
                # Compute gradient for kernel[u, v, :, :] as sum_{n,i,j} patch * delta.
                # This is achieved by contracting over batch and spatial dimensions.
                kernel_gradient[u, v, :, :] = torch.einsum('nhwc,nhwo->co', patch, delta)

        # Compute gradient with respect to the inputs.
        inputs_gradient = torch.zeros_like(inputs)
        for u in range(K):
            for v in range(K):
                # Compute contribution for each kernel position:
                # For each patch in delta, propagate back through the kernel.
                patch_grad = torch.einsum('nhwo,co->nhwc', delta, self._kernel[u, v, :, :])
                # Add the gradient to the appropriate slice of inputs_gradient.
                inputs_gradient[:, u:u + s * H_out:s, v:v + s * W_out:s, :] += patch_grad
        
        
        # If requested, verify that the three computed gradients are correct.
        if self._verify:
            with torch.enable_grad():
                inputs.requires_grad_(True)
                inputs.grad = self._kernel.grad = self._bias.grad = None
                reference = (outputs > 0) * torch.nn.functional.conv2d(
                    inputs.movedim(-1, 1), self._kernel.permute(3, 2, 0, 1), self._bias, self._stride).movedim(1, -1)
                reference.backward(gradient=outputs_gradient, inputs=[inputs, self._kernel, self._bias])
                for name, computed, reference in zip(
                        ["Bias", "Kernel", "Inputs"], [bias_gradient, kernel_gradient, inputs_gradient],
                        [self._bias.grad, self._kernel.grad, inputs.grad]):
                    np.testing.assert_allclose(computed.detach().numpy(), reference.detach().numpy(),
                                               atol=2e-4, err_msg=name + " gradient differs!")

        # Return the inputs gradient, the layer variables, and their gradients.
        return inputs_gradient, [self._kernel, self._bias], [kernel_gradient, bias_gradient]


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

        # Create the convolutional layers according to `args.cnn`.
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convs = []
        for layer in args.cnn.split(","):
            filters, kernel_size, stride = map(int, layer.split("-"))
            self._convs.append(Convolution(filters, kernel_size, stride, input_shape, args.verify))
            input_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, filters]

        # Create the classification head.
        self._classifier = torch.nn.Linear(np.prod(input_shape), MNIST.LABELS)

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for batch in dataset.batches(self._args.batch_size, shuffle=True):
            # Forward pass through the convolutions.
            hidden = batch["images"].to(torch.float32).movedim(1, -1) / 255
            conv_values = [hidden]
            for conv in self._convs:
                hidden = conv.forward(hidden)
                conv_values.append(hidden)

            # Run the classification head.
            hidden_flat = torch.flatten(hidden, 1)
            predictions = self._classifier(hidden_flat).softmax(dim=-1)

            # Compute the gradients of the classifier and the convolution output.
            one_hot_labels = torch.nn.functional.one_hot(batch["labels"].to(torch.int64), MNIST.LABELS)
            d_logits = (predictions - one_hot_labels) / len(batch["images"])
            variables = [self._classifier.bias, self._classifier.weight]
            gradients = [d_logits.sum(dim=0), d_logits.T @ hidden_flat]
            hidden_gradient = (d_logits @ self._classifier.weight).reshape(hidden.shape)

            # Backpropagate the gradient through the convolutions.
            for conv, inputs, outputs in reversed(list(zip(self._convs, conv_values[:-1], conv_values[1:]))):
                hidden_gradient, conv_variables, conv_gradients = conv.backward(inputs, outputs, hidden_gradient)
                variables.extend(conv_variables)
                gradients.extend(conv_gradients)

            # Update the weights using a manual SGD.
            for variable, gradient in zip(variables, gradients):
                variable -= self._args.learning_rate * gradient

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        total = correct = 0
        for batch in dataset.batches(self._args.batch_size):
            hidden = batch["images"].to(torch.float32).movedim(1, -1) / 255
            for conv in self._convs:
                hidden = conv.forward(hidden)
            hidden = torch.flatten(hidden, 1)
            predictions = self._classifier(hidden)
            correct += torch.sum(predictions.argmax(dim=-1) == batch["labels"])
            total += len(batch["labels"])
        return correct / total


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Do not compute gradients in this assignment.
    torch.set_grad_enabled(False)

    # Load data, using only 5 000 training images, and create the dataloaders.
    mnist = MNIST(sizes={"train": 5_000})

    # Create the model.
    model = Model(args)

    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)

        dev_accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * dev_accuracy))

    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy))

    # Return dev and test accuracies for ReCodEx to validate.
    return dev_accuracy, test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
