import pandas as pd
import plotly.express as px
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
    tanh = nn.Tanh()
    softmax = nn.Softmax()
    celu = nn.CELU()

    X = torch.linspace(-5, 5, 100)
    y_gelu, y_relu = gelu(X), relu(X)
    y_silu = silu(X)
    y_tanh = tanh(X)
    y_celu = celu(X)
    y_softmax = softmax(X)

    df = pd.DataFrame(
        {
            "Input": X.numpy(),
            "GELU": y_gelu.numpy(),
            "ReLU": y_relu.numpy(),
            "SiLU": y_silu.numpy(),
            "Tanh": y_tanh.numpy(),
            "Softmax": y_softmax.numpy(),
            "CELU": y_celu.numpy(),
        }
    )

    # Plot using plotly
    fig = px.line(
        df,
        y=["GELU", "ReLU", "SiLU", "Tanh", "Softmax", "CELU"],
        x="Input",
        title="Activation Functions: GELU vs ReLU vs SiLU vs Tanh vs Softmax vs CELU",
        labels={"value": "Activation Output", "Input": "Input Value"},
    )
    fig.show()
