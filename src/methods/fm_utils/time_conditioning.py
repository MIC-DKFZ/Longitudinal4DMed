from torch import nn


class TemporalFiLMAdapter(nn.Module):
    """Basic FiLM time-conditioning: embeds a continuous time value and uses
    it to scale/shift a feature map. This is the one time-conditioning
    contract shared by every backbone in fm_utils.backbones (ConditionedUNet,
    NanoDiT's adaLN is the transformer-block analogue of the same idea, and
    ConvLSTMBackbone reuses this class directly).
    """

    def __init__(self, time_embed_dim, hidden_dim, feature_dim):
        super().__init__()
        self.film_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feature_dim)  # scale + shift
        )

    def forward(self, h, t_emb):
        """
        h:        [B, C, ...] feature map
        t_emb:    [B, D] temporal embedding (e.g., Gaussian Fourier, sinusoidal, or raw scalar time)
        returns:  FiLM-modulated feature map
        """
        gamma_beta = self.film_mlp(t_emb)  # [B, 2*C]
        gamma, beta = gamma_beta.chunk(2, dim=1)  # [B, C], [B, C]

        while gamma.dim() < h.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        return h * (1 + gamma) + beta
