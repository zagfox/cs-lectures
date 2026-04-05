from typing import Dict, List

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad
from transformer import linear_layer, single_head_attention, encoder_layer


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


def test_single_head_attention_output_shape():
    """Output shape should be (batch, seq, model_dim)."""
    batch, seq, model_dim = 2, 4, 8
    X = ad.Variable("X")
    W_Q, W_K, W_V = ad.Variable("W_Q"), ad.Variable("W_K"), ad.Variable("W_V")
    y = single_head_attention(X, W_Q, W_K, W_V, model_dim)
    evaluator = ad.Evaluator([y])

    X_val = torch.randn(batch, seq, model_dim, dtype=torch.float64)
    W_val = torch.randn(model_dim, model_dim, dtype=torch.float64)

    result = evaluator.run({X: X_val, W_Q: W_val, W_K: W_val.clone(), W_V: W_val.clone()})
    assert result[0].shape == (batch, seq, model_dim)


def test_single_head_attention_matches_pytorch():
    """Forward output matches manual PyTorch computation."""
    import math
    batch, seq, model_dim = 2, 3, 4
    X = ad.Variable("X")
    W_Q, W_K, W_V = ad.Variable("W_Q"), ad.Variable("W_K"), ad.Variable("W_V")
    y = single_head_attention(X, W_Q, W_K, W_V, model_dim)
    evaluator = ad.Evaluator([y])

    torch.manual_seed(0)
    X_val = torch.randn(batch, seq, model_dim, dtype=torch.float64)
    WQ_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WK_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WV_val = torch.randn(model_dim, model_dim, dtype=torch.float64)

    Q = X_val @ WQ_val
    K = X_val @ WK_val
    V = X_val @ WV_val
    scores = Q @ K.transpose(-2, -1) / math.sqrt(model_dim)
    A = torch.softmax(scores, dim=-1)
    expected = A @ V

    check_evaluator_output(
        evaluator,
        {X: X_val, W_Q: WQ_val, W_K: WK_val, W_V: WV_val},
        [expected],
    )


def test_encoder_layer_output_shape():
    """Output shape should be (batch, seq, model_dim)."""
    batch, seq, model_dim = 2, 4, 8
    eps = 1e-5
    X = ad.Variable("X")
    W_Q, W_K, W_V = ad.Variable("W_Q"), ad.Variable("W_K"), ad.Variable("W_V")
    W_O = ad.Variable("W_O")
    W_1, b_1 = ad.Variable("W_1"), ad.Variable("b_1")
    y = encoder_layer(X, W_Q, W_K, W_V, W_O, W_1, b_1, model_dim, eps)
    evaluator = ad.Evaluator([y])

    torch.manual_seed(0)
    vals = {
        X:   torch.randn(batch, seq, model_dim, dtype=torch.float64),
        W_Q: torch.randn(model_dim, model_dim, dtype=torch.float64),
        W_K: torch.randn(model_dim, model_dim, dtype=torch.float64),
        W_V: torch.randn(model_dim, model_dim, dtype=torch.float64),
        W_O: torch.randn(model_dim, model_dim, dtype=torch.float64),
        W_1: torch.randn(model_dim, model_dim, dtype=torch.float64),
        b_1: torch.randn(model_dim, dtype=torch.float64),
    }
    result = evaluator.run(vals)
    assert result[0].shape == (batch, seq, model_dim)


def test_encoder_layer_matches_pytorch():
    """Forward output matches manual PyTorch computation."""
    import math
    batch, seq, model_dim = 2, 3, 8
    eps = 1e-5
    X = ad.Variable("X")
    W_Q, W_K, W_V = ad.Variable("W_Q"), ad.Variable("W_K"), ad.Variable("W_V")
    W_O = ad.Variable("W_O")
    W_1, b_1 = ad.Variable("W_1"), ad.Variable("b_1")
    y = encoder_layer(X, W_Q, W_K, W_V, W_O, W_1, b_1, model_dim, eps)
    evaluator = ad.Evaluator([y])

    torch.manual_seed(3)
    X_val  = torch.randn(batch, seq, model_dim, dtype=torch.float64)
    WQ_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WK_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WV_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WO_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    W1_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    b1_val = torch.randn(model_dim, dtype=torch.float64)

    # Reference computation
    Q = X_val @ WQ_val
    K = X_val @ WK_val
    V = X_val @ WV_val
    A = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(model_dim), dim=-1)
    attn_proj = (A @ V) @ WO_val
    ln1 = torch.layer_norm(attn_proj, [model_dim], eps=eps)
    ff = torch.relu(ln1 @ W1_val + b1_val)
    expected = torch.layer_norm(ff, [model_dim], eps=eps)

    check_evaluator_output(
        evaluator,
        {X: X_val, W_Q: WQ_val, W_K: WK_val, W_V: WV_val,
         W_O: WO_val, W_1: W1_val, b_1: b1_val},
        [expected],
    )


if __name__ == "__main__":
    test_linear_layer_2d()
    test_linear_layer_3d()
    test_linear_layer_zero_bias()
    test_single_head_attention_output_shape()
    test_single_head_attention_matches_pytorch()
    test_encoder_layer_output_shape()
    test_encoder_layer_matches_pytorch()
