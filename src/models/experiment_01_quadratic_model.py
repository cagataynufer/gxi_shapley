import torch
import torch.nn as nn



# Quadratic model with sparse pairwise interactions
class QuadraticModel(nn.Module):
    """
    Simple 5-feature quadratic model with only two interactions:

        f(x) = a0*x0 + a1*x1 + a2*x2 + a3*x3 + a4*x4
               + b01*x0*x1 + b03*x0*x3

    Used for:
    - analytical validation
    - off-diagonal zero checks
    - coalition-path experiments
    """

    def __init__(self):
        super().__init__()

        # linear coefficients
        self.a0 = 1.0
        self.a1 = -0.5
        self.a2 = 0.0
        self.a3 = 0.3
        self.a4 = -0.2

        # interaction coefficients
        self.b01 = 2.0
        self.b03 = -1.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 5)
        """
        x0, x1, x2, x3, x4 = x.unbind(-1)

        linear_part = (
            self.a0 * x0 +
            self.a1 * x1 +
            self.a2 * x2 +
            self.a3 * x3 +
            self.a4 * x4
        )

        interaction_part = (
            self.b01 * x0 * x1 +
            self.b03 * x0 * x3
        )

        return linear_part + interaction_part


