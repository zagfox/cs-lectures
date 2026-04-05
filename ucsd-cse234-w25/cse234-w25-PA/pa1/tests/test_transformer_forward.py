from typing import Dict, List

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad
from transformer import linear_layer


def check_evaluator_output(
    evaluator: ad.Evaluator,
    input_values: Dict[ad.Node, torch.Tensor],
    expected_outputs: List[torch.Tensor],
) -> None:
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        torch.testing.assert_close(output_val, expected_val, atol=1e-4, rtol=1e-4)


def test_linear_layer_2d():
    """2D input (batch, in_features): output = X @ W + b."""
    X = ad.Variable("X")
    W = ad.Variable("W")
    b = ad.Variable("b")
    y = linear_layer(X, W, b)
    evaluator = ad.Evaluator([y])

    X_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    W_val = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    b_val = torch.tensor([0.5, -0.5], dtype=torch.float64)

    check_evaluator_output(
        evaluator,
        {X: X_val, W: W_val, b: b_val},
        [X_val @ W_val + b_val],
    )


def test_linear_layer_3d():
    """3D input (batch, seq, in_features): bias broadcasts over batch and seq dims."""
    X = ad.Variable("X")
    W = ad.Variable("W")
    b = ad.Variable("b")
    y = linear_layer(X, W, b)
    evaluator = ad.Evaluator([y])

    X_val = torch.randn(2, 4, 3, dtype=torch.float64)
    W_val = torch.randn(3, 5, dtype=torch.float64)
    b_val = torch.randn(5, dtype=torch.float64)

    check_evaluator_output(
        evaluator,
        {X: X_val, W: W_val, b: b_val},
        [X_val @ W_val + b_val],
    )


def test_linear_layer_zero_bias():
    """With zero bias, output equals X @ W."""
    X = ad.Variable("X")
    W = ad.Variable("W")
    b = ad.Variable("b")
    y = linear_layer(X, W, b)
    evaluator = ad.Evaluator([y])

    X_val = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    W_val = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64)
    b_val = torch.zeros(2, dtype=torch.float64)

    check_evaluator_output(
        evaluator,
        {X: X_val, W: W_val, b: b_val},
        [X_val @ W_val],
    )


if __name__ == "__main__":
    test_linear_layer_2d()
    test_linear_layer_3d()
    test_linear_layer_zero_bias()
