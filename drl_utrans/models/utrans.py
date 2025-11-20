import torch
import torch.nn as nn
from typing import Tuple


class _NormLinearReLU(nn.Module):
    """
    A helper block consisting of LayerNorm -> Linear -> ReLU.
    
    This block corresponds to the red arrows in the architecture diagram.
    
    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, elementwise_affine=True)
        self.fc   = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, in_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, out_dim).
        """
        # x: (batch, seq_len, in_dim)
        x = self.ln(x)  # (B, T, F)
        return self.relu(self.fc(x))


class UTransNet(nn.Module):
    """
    U-Net-style encoder/decoder with a Transformer bottleneck for financial time-series.

    Architecture:
    ── Input  ➜  1×64  ➜  1×128  ➜  1×256  ➜  Transformer  ─┐
                     ↑        ↑         ↑                 │
                     └────────┴─────────┴─────────────────┘  (skip-adds)
                               ↓         ↓         ↓
                          1×256 ▸ 1×128 ▸ 1×64 ▸ heads

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    n_actions : int, optional
        Number of discrete actions (e.g., Buy, Sell, Hold), by default 3.
    n_transformer_heads : int, optional
        Number of attention heads in the Transformer bottleneck, by default 8.
    n_transformer_layers : int, optional
        Number of Transformer encoder layers, by default 1.
    """
    def __init__(
        self,
        input_dim: int,
        n_actions: int = 3,
        n_transformer_heads: int = 8,
        n_transformer_layers: int = 1,
    ):
        super().__init__()

        # --- Encoder ---------------------------------------------------------
        self.enc1 = _NormLinearReLU(input_dim, 64)    # 1 × 64
        self.enc2 = _NormLinearReLU(64, 128)          # 1 × 128
        self.enc3 = _NormLinearReLU(128, 256)         # 1 × 256

        # --- Transformer bottleneck -----------------------------------------
        t_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=n_transformer_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            t_layer, num_layers=n_transformer_layers
        )

        # --- Decoder (add-skip ➜ Norm-Linear-ReLU) ---------------------------
        self.dec1 = _NormLinearReLU(256, 128)         # 1 × 128
        self.dec2 = _NormLinearReLU(128, 64)          # 1 × 64

        # --- Dual heads ------------------------------------------------------
        # global average over the sequence length → 1 × 64 vector
        self.act_head   = nn.Linear(65, n_actions)    # categorical action
        self.weight_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()                              # 0-1 weighting
        )

    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the UTransNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - q_values: Tensor of shape (batch, n_actions) representing action logits.
            - act_weight: Tensor of shape (batch,) representing position sizing confidence (0-1).
        """
        # Encoder -------------------------------------------------------------
        e1 = self.enc1(x)     # 1 × 64
        e2 = self.enc2(e1)    # 1 × 128
        e3 = self.enc3(e2)    # 1 × 256

        # Transformer ---------------------------------------------------------
        t = self.transformer(e3)  # (batch, seq_len, 256)

        # Decoder with additive skips ----------------------------------------
        d1_in = t + e3            # add 1 × 256 skip
        d1    = self.dec1(d1_in)  # → 1 × 128

        d2_in = d1 + e2           # add 1 × 128 skip
        d2    = self.dec2(d2_in)  # → 1 × 64

        final = d2 + e1           # last skip (no further reduction)

        # Heads ---------------------------------------------------------------
        pooled = final.mean(dim=1)  # global average over sequence
        act_weight = self.weight_head(pooled).squeeze(-1)
        
        # Condition action head on the weight
        pooled_with_weight = torch.cat([pooled, act_weight.unsqueeze(1)], dim=1)  # (batch, 65)
        q_values   = self.act_head(pooled_with_weight)

        return q_values, act_weight
