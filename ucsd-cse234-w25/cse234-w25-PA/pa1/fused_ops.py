from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        matmul_result = torch.matmul(input_values[0], input_values[1])
        return torch.layer_norm(
            matmul_result,
            normalized_shape=node.attrs["normalized_shape"],
            eps=node.attrs["eps"],
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        A, B = node.inputs[0], node.inputs[1]
        # Reconstruct sub-graph: C = A @ B, Y_ln = LayerNorm(C)
        C = matmul(A, B)
        Y_ln = layernorm(C, node.attrs["normalized_shape"], node.attrs["eps"])
        # Gradient through LayerNorm: dL/dC (symbolic, uses mean/sqrt/sum_op ops)
        dC = layernorm.gradient(Y_ln, output_grad)[0]
        # Gradient through MatMul: dL/dA and dL/dB
        return matmul.gradient(C, dC)


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        matmul_result = torch.matmul(input_values[0], input_values[1])
        return torch.softmax(matmul_result, dim=node.attrs["dim"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        A, B = node.inputs[0], node.inputs[1]
        # Reconstruct sub-graph: C = A @ B, S = softmax(C)
        C = matmul(A, B)
        S = softmax(C, dim=node.attrs["dim"])
        # Gradient through softmax: dL/dC
        dC = softmax.gradient(S, output_grad)[0]
        # Gradient through MatMul: dL/dA and dL/dB
        return matmul.gradient(C, dC)

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()