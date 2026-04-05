import functools
import math
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28


def linear_layer(X: ad.Node, W: ad.Node, b: ad.Node) -> ad.Node:
    """Linear transformation: output = X @ W + b

    Parameters
    ----------
    X: ad.Node
        Input node, shape (..., in_features).
    W: ad.Node
        Weight node, shape (in_features, out_features).
    b: ad.Node
        Bias node, shape (out_features,).

    Returns
    -------
    output: ad.Node
        Output node, shape (..., out_features).
    """
    return ad.matmul(X, W) + b


def single_head_attention(
    X: ad.Node, W_Q: ad.Node, W_K: ad.Node, W_V: ad.Node, model_dim: int
) -> ad.Node:
    """Single-head scaled dot-product attention.

    Parameters
    ----------
    X: ad.Node
        Input node, shape (batch_size, seq_length, model_dim).
    W_Q, W_K, W_V: ad.Node
        Projection weight nodes, each shape (model_dim, model_dim).
    model_dim: int
        Dimension d_k used for scaling.

    Returns
    -------
    output: ad.Node
        Attention output, shape (batch_size, seq_length, model_dim).
    """
    Q = ad.matmul(X, W_Q)                          # (batch, seq, model_dim)
    K = ad.matmul(X, W_K)                          # (batch, seq, model_dim)
    V = ad.matmul(X, W_V)                          # (batch, seq, model_dim)

    K_T = ad.transpose(K, -2, -1)                  # (batch, model_dim, seq)
    scores = ad.matmul(Q, K_T)                     # (batch, seq, seq)
    scores_scaled = ad.mul_by_const(scores, 1.0 / math.sqrt(model_dim))
    A = ad.softmax(scores_scaled, dim=-1)          # (batch, seq, seq)

    return ad.matmul(A, V)                         # (batch, seq, model_dim)


def encoder_layer(
    X: ad.Node,
    W_Q: ad.Node, W_K: ad.Node, W_V: ad.Node,
    W_O: ad.Node,
    W_1: ad.Node, b_1: ad.Node,
    model_dim: int,
    eps: float,
) -> ad.Node:
    """Single encoder layer: self-attention + output projection + LayerNorm + FFN + LayerNorm.

    Parameters
    ----------
    X: ad.Node
        Input, shape (batch_size, seq_length, model_dim).
    W_Q, W_K, W_V: ad.Node
        Attention projection weights, each (model_dim, model_dim).
    W_O: ad.Node
        Output projection weight, shape (model_dim, model_dim). No bias.
    W_1: ad.Node
        FFN first-layer weight, shape (model_dim, model_dim).
    b_1: ad.Node
        FFN first-layer bias, shape (model_dim,).
    model_dim: int
        Hidden dimension (also used as d_k for attention scaling).
    eps: float
        LayerNorm epsilon.

    Returns
    -------
    output: ad.Node
        Shape (batch_size, seq_length, model_dim).
    """
    # Self-attention then output projection
    attn = single_head_attention(X, W_Q, W_K, W_V, model_dim)  # (batch, seq, model_dim)
    attn_proj = ad.matmul(attn, W_O)                            # (batch, seq, model_dim)
    ln1 = ad.layernorm(attn_proj, [model_dim], eps)             # (batch, seq, model_dim)

    # Position-wise feed-forward: Linear → ReLU → LayerNorm
    ff = ad.relu(linear_layer(ln1, W_1, b_1))                   # (batch, seq, model_dim)
    ln2 = ad.layernorm(ff, [model_dim], eps)                    # (batch, seq, model_dim)

    return ln2


def transformer(X: ad.Node, nodes: List[ad.Node],
                      model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes

    # Encoder layer → (batch, seq, model_dim)
    enc = encoder_layer(X, W_Q, W_K, W_V, W_O, W_1, b_1, model_dim, eps)

    # Mean-pool over sequence → (batch, model_dim)
    pooled = ad.mean(enc, dim=1, keepdim=False)

    # Classification head → (batch, num_classes)
    return linear_layer(pooled, W_2, b_2)


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    log_probs = ad.log(ad.softmax(Z, dim=-1))          # (batch, num_classes)
    per_sample = ad.sum_op(y_one_hot * log_probs, dim=1) # (batch,)
    return ad.sum_op(per_sample, dim=0) * (-1.0 / batch_size)  # scalar ()


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size> num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        
        # Compute forward and backward passes
        # f_run_model returns [y_predict, loss, grad_W_Q, grad_W_K, grad_W_V,
        #                       grad_W_O, grad_W_1, grad_W_2, grad_b_1, grad_b_2]
        result = f_run_model(X_batch, y_batch, model_weights)
        _, loss_val, grad_W_Q, grad_W_K, grad_W_V, grad_W_O, grad_W_1, grad_W_2, grad_b_1, grad_b_2 = result

        # Update weights and biases
        # 3-D weight grads (batch, in, out) → sum over batch; 2-D grad_W_2 is already correct
        W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = model_weights
        W_Q -= lr * grad_W_Q.sum(dim=0)
        W_K -= lr * grad_W_K.sum(dim=0)
        W_V -= lr * grad_W_V.sum(dim=0)
        W_O -= lr * grad_W_O.sum(dim=0)
        W_1 -= lr * grad_W_1.sum(dim=0)
        W_2 -= lr * grad_W_2                    # pooled input is 2-D; grad already (in, out)
        b_1 -= lr * grad_b_1.sum(dim=(0, 1))    # grad shape (batch, seq, model_dim)
        b_2 -= lr * grad_b_2.sum(dim=0)         # grad shape (batch, num_classes)
        model_weights = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

        # Accumulate the loss (loss_val is mean over batch; scale back to sum)
        total_loss += loss_val.item() * batch_size


    # Compute the average loss
    
    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)

    # TODO: Your code here
    # You should return the list of parameters and the loss
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5 

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # Define the forward graph.
    X_node = ad.Variable(name="X")
    W_Q_node = ad.Variable(name="W_Q")
    W_K_node = ad.Variable(name="W_K")
    W_V_node = ad.Variable(name="W_V")
    W_O_node = ad.Variable(name="W_O")
    W_1_node = ad.Variable(name="W_1")
    W_2_node = ad.Variable(name="W_2")
    b_1_node = ad.Variable(name="b_1")
    b_2_node = ad.Variable(name="b_2")
    weight_nodes = [W_Q_node, W_K_node, W_V_node, W_O_node, W_1_node, W_2_node, b_1_node, b_2_node]

    y_predict: ad.Node = transformer(X_node, weight_nodes, model_dim, seq_length, eps, batch_size, num_classes)
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)

    # Construct the backward graph.
    grads: List[ad.Node] = ad.gradients(loss, weight_nodes)
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(X_batch, y_batch, model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = model_weights
        result = evaluator.run(
            input_values={
                X_node: X_batch.double(),
                y_groundtruth: y_batch.double(),
                W_Q_node: W_Q, W_K_node: W_K, W_V_node: W_V, W_O_node: W_O,
                W_1_node: W_1, W_2_node: W_2, b_1_node: b_1, b_2_node: b_2,
            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size> num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = model_weights
            logits = test_evaluator.run({
                X_node: X_batch.double(),
                W_Q_node: W_Q, W_K_node: W_K, W_V_node: W_V, W_O_node: W_O,
                W_1_node: W_1, W_2_node: W_2, b_1_node: b_1, b_2_node: b_2,
            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [
        torch.DoubleTensor(W_Q_val), torch.DoubleTensor(W_K_val),
        torch.DoubleTensor(W_V_val), torch.DoubleTensor(W_O_val),
        torch.DoubleTensor(W_1_val), torch.DoubleTensor(W_2_val),
        torch.DoubleTensor(b_1_val), torch.DoubleTensor(b_2_val),
    ]
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
