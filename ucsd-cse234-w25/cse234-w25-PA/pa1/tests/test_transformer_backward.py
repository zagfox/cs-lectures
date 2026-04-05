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


def test_linear_layer_grad_X():
    """Gradient w.r.t. X matches PyTorch autograd."""
    X = ad.Variable("X")
    W = ad.Variable("W")
    b = ad.Variable("b")
    y = linear_layer(X, W, b)
    [grad_X] = ad.gradients(y, [X])
    evaluator = ad.Evaluator([grad_X])

    X_val = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    W_val = torch.tensor([[0.5, -0.5], [1.0, 2.0]], dtype=torch.float64)
    b_val = torch.tensor([0.1, 0.2], dtype=torch.float64)

    X_t = X_val.clone().requires_grad_(True)
    (X_t @ W_val + b_val).sum().backward()

    check_evaluator_output(
        evaluator,
        {X: X_val, W: W_val, b: b_val},
        [X_t.grad],
    )


def test_linear_layer_grad_W():
    """Gradient w.r.t. W matches PyTorch autograd."""
    X = ad.Variable("X")
    W = ad.Variable("W")
    b = ad.Variable("b")
    y = linear_layer(X, W, b)
    [grad_W] = ad.gradients(y, [W])
    evaluator = ad.Evaluator([grad_W])

    X_val = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    W_val = torch.tensor([[0.5, -0.5], [1.0, 2.0]], dtype=torch.float64)
    b_val = torch.tensor([0.1, 0.2], dtype=torch.float64)

    W_t = W_val.clone().requires_grad_(True)
    (X_val @ W_t + b_val).sum().backward()

    check_evaluator_output(
        evaluator,
        {X: X_val, W: W_val, b: b_val},
        [W_t.grad],
    )


def test_linear_layer_grad_b():
    """Gradient w.r.t. b matches PyTorch autograd (summed over batch)."""
    X = ad.Variable("X")
    W = ad.Variable("W")
    b = ad.Variable("b")
    y = linear_layer(X, W, b)
    [grad_b] = ad.gradients(y, [b])
    evaluator = ad.Evaluator([grad_b])

    X_val = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    W_val = torch.tensor([[0.5, -0.5], [1.0, 2.0]], dtype=torch.float64)
    b_val = torch.tensor([0.1, 0.2], dtype=torch.float64)

    b_t = b_val.clone().requires_grad_(True)
    (X_val @ W_val + b_t).sum().backward()

    # grad_b from auto_diff has shape (batch, out); sum over batch to match PyTorch
    result = evaluator.run({X: X_val, W: W_val, b: b_val})
    grad_b_val = result[0].sum(dim=0)
    torch.testing.assert_close(grad_b_val, b_t.grad, atol=1e-4, rtol=1e-4)


def test_linear_layer_grad_3d():
    """Gradients w.r.t. X and W with 3D input match PyTorch autograd.

    auto_diff returns grad_W with shape (batch, in, out) for 3D inputs;
    summing over dim=0 gives the same result as PyTorch's accumulated gradient.
    """
    X = ad.Variable("X")
    W = ad.Variable("W")
    b = ad.Variable("b")
    y = linear_layer(X, W, b)
    grad_X, grad_W = ad.gradients(y, [X, W])
    evaluator = ad.Evaluator([grad_X, grad_W])

    X_val = torch.randn(2, 4, 3, dtype=torch.float64)
    W_val = torch.randn(3, 5, dtype=torch.float64)
    b_val = torch.randn(5, dtype=torch.float64)

    X_t = X_val.clone().requires_grad_(True)
    W_t = W_val.clone().requires_grad_(True)
    (X_t @ W_t + b_val).sum().backward()

    grad_X_val, grad_W_val = evaluator.run({X: X_val, W: W_val, b: b_val})
    # grad_X shape matches (batch, seq, in)
    torch.testing.assert_close(grad_X_val, X_t.grad, atol=1e-4, rtol=1e-4)
    # grad_W shape is (batch, in, out); sum over batch to match PyTorch
    torch.testing.assert_close(grad_W_val.sum(dim=0), W_t.grad, atol=1e-4, rtol=1e-4)


def test_single_head_attention_grad_W_Q():
    """Gradient w.r.t. W_Q matches PyTorch autograd (summed over batch)."""
    import math
    batch, seq, model_dim = 2, 3, 4
    X = ad.Variable("X")
    W_Q, W_K, W_V = ad.Variable("W_Q"), ad.Variable("W_K"), ad.Variable("W_V")
    y = single_head_attention(X, W_Q, W_K, W_V, model_dim)
    [grad_WQ] = ad.gradients(y, [W_Q])
    evaluator = ad.Evaluator([grad_WQ])

    torch.manual_seed(1)
    X_val = torch.randn(batch, seq, model_dim, dtype=torch.float64)
    WQ_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WK_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WV_val = torch.randn(model_dim, model_dim, dtype=torch.float64)

    WQ_t = WQ_val.clone().requires_grad_(True)
    Q = X_val @ WQ_t
    K = X_val @ WK_val
    V = X_val @ WV_val
    A = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(model_dim), dim=-1)
    (A @ V).sum().backward()

    # auto_diff returns grad shape (batch, model_dim, model_dim); sum over batch
    result = evaluator.run({X: X_val, W_Q: WQ_val, W_K: WK_val, W_V: WV_val})
    torch.testing.assert_close(result[0].sum(dim=0), WQ_t.grad, atol=1e-4, rtol=1e-4)


def test_single_head_attention_grad_X():
    """Gradient w.r.t. X matches PyTorch autograd."""
    import math
    batch, seq, model_dim = 2, 3, 4
    X = ad.Variable("X")
    W_Q, W_K, W_V = ad.Variable("W_Q"), ad.Variable("W_K"), ad.Variable("W_V")
    y = single_head_attention(X, W_Q, W_K, W_V, model_dim)
    [grad_X] = ad.gradients(y, [X])
    evaluator = ad.Evaluator([grad_X])

    torch.manual_seed(2)
    X_val = torch.randn(batch, seq, model_dim, dtype=torch.float64)
    WQ_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WK_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WV_val = torch.randn(model_dim, model_dim, dtype=torch.float64)

    X_t = X_val.clone().requires_grad_(True)
    Q = X_t @ WQ_val
    K = X_t @ WK_val
    V = X_t @ WV_val
    A = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(model_dim), dim=-1)
    (A @ V).sum().backward()

    check_evaluator_output(
        evaluator,
        {X: X_val, W_Q: WQ_val, W_K: WK_val, W_V: WV_val},
        [X_t.grad],
    )


def test_encoder_layer_grad_X():
    """Gradient w.r.t. X through the full encoder layer matches PyTorch autograd."""
    import math
    batch, seq, model_dim = 2, 3, 8
    eps = 1e-5
    X = ad.Variable("X")
    W_Q, W_K, W_V = ad.Variable("W_Q"), ad.Variable("W_K"), ad.Variable("W_V")
    W_O = ad.Variable("W_O")
    W_1, b_1 = ad.Variable("W_1"), ad.Variable("b_1")
    y = encoder_layer(X, W_Q, W_K, W_V, W_O, W_1, b_1, model_dim, eps)
    [grad_X] = ad.gradients(y, [X])
    evaluator = ad.Evaluator([grad_X])

    torch.manual_seed(4)
    X_val  = torch.randn(batch, seq, model_dim, dtype=torch.float64)
    WQ_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WK_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WV_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WO_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    W1_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    b1_val = torch.randn(model_dim, dtype=torch.float64)

    X_t = X_val.clone().requires_grad_(True)
    Q = X_t @ WQ_val
    K = X_t @ WK_val
    V = X_t @ WV_val
    A = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(model_dim), dim=-1)
    attn_proj = (A @ V) @ WO_val
    ln1 = torch.layer_norm(attn_proj, [model_dim], eps=eps)
    ff = torch.relu(ln1 @ W1_val + b1_val)
    torch.layer_norm(ff, [model_dim], eps=eps).sum().backward()

    check_evaluator_output(
        evaluator,
        {X: X_val, W_Q: WQ_val, W_K: WK_val, W_V: WV_val,
         W_O: WO_val, W_1: W1_val, b_1: b1_val},
        [X_t.grad],
    )


def test_encoder_layer_grad_W_O():
    """Gradient w.r.t. W_O matches PyTorch autograd (summed over batch)."""
    import math
    batch, seq, model_dim = 2, 3, 8
    eps = 1e-5
    X = ad.Variable("X")
    W_Q, W_K, W_V = ad.Variable("W_Q"), ad.Variable("W_K"), ad.Variable("W_V")
    W_O = ad.Variable("W_O")
    W_1, b_1 = ad.Variable("W_1"), ad.Variable("b_1")
    y = encoder_layer(X, W_Q, W_K, W_V, W_O, W_1, b_1, model_dim, eps)
    [grad_WO] = ad.gradients(y, [W_O])
    evaluator = ad.Evaluator([grad_WO])

    torch.manual_seed(5)
    X_val  = torch.randn(batch, seq, model_dim, dtype=torch.float64)
    WQ_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WK_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WV_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    WO_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    W1_val = torch.randn(model_dim, model_dim, dtype=torch.float64)
    b1_val = torch.randn(model_dim, dtype=torch.float64)

    WO_t = WO_val.clone().requires_grad_(True)
    Q = X_val @ WQ_val
    K = X_val @ WK_val
    V = X_val @ WV_val
    A = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(model_dim), dim=-1)
    attn_proj = (A @ V) @ WO_t
    ln1 = torch.layer_norm(attn_proj, [model_dim], eps=eps)
    ff = torch.relu(ln1 @ W1_val + b1_val)
    torch.layer_norm(ff, [model_dim], eps=eps).sum().backward()

    # auto_diff returns grad shape (batch, model_dim, model_dim); sum over batch
    result = evaluator.run({X: X_val, W_Q: WQ_val, W_K: WK_val, W_V: WV_val,
                            W_O: WO_val, W_1: W1_val, b_1: b1_val})
    torch.testing.assert_close(result[0].sum(dim=0), WO_t.grad, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_linear_layer_grad_X()
    test_linear_layer_grad_W()
    test_linear_layer_grad_b()
    test_linear_layer_grad_3d()
    test_single_head_attention_grad_W_Q()
    test_single_head_attention_grad_X()
    test_encoder_layer_grad_X()
    test_encoder_layer_grad_W_O()
