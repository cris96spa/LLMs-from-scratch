import matplotlib.pyplot as plt
import torch
from torch import nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


if __name__ == "__main__":
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    gelu = GELU()
    relu = nn.ReLU()
    silu = nn.SiLU()

    X = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(X), relu(X)
    y_silu = silu(X)
    plt.figure(figsize=(10, 5))
    plt.plot(X, y_gelu, label="GELU", color="blue")
    plt.plot(X, y_relu, label="ReLU", color="red")
    plt.plot(X, y_silu, label="SiLU", color="orange")
    plt.title("GELU vs ReLU Activation Functions")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid()
    plt.show()
