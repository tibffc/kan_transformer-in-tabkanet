from typing import Dict, List, Optional, Literal
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
from torch.nn import BatchNorm1d
from .base_blocks import Activation, ColumnEmbedding
import torch.nn.init as init
from torch import Tensor 

# ============== Вспомогательные классы ==============

class CatEncoder(nn.Module):
    def __init__(self, vocabulary: Dict[str, Dict[str, int]], embedding_dim: int):
        super(CatEncoder, self).__init__()
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.columnembedding = ColumnEmbedding(vocabulary, embedding_dim)

    def forward(self, x: torch.Tensor, continuous_x_res: torch.Tensor):
        batch_size = x.size(0)
        x = [self.columnembedding(x[:, i], col) for i, col in enumerate(self.vocabulary)]
        x = torch.stack(x, dim=1)
        x = torch.cat((x, continuous_x_res), dim=1)
        return x


class NumEncoder(nn.Module):
    def __init__(self, num_features: int):
        super(NumEncoder, self).__init__()
        self.norm = nn.LayerNorm([num_features])

    def forward(self, x: torch.Tensor):
        return self.norm(x)


class NoiseLearnModule(nn.Module):
    def __init__(self, feature_dim, n_bins=4, noise_scale=0.1):
        super(NoiseLearnModule, self).__init__()
        self.noise_params = nn.Parameter(torch.zeros(feature_dim * n_bins))
        self.noise_scale = noise_scale

    def forward(self, x, bins):
        batch_size, feature_dim = x.shape
        noise_params = self.noise_params.view(feature_dim, -1)
        noise = torch.zeros_like(x)

        for i in range(feature_dim):
            feature_values = x[:, i]
            for j in range(len(bins[i]) - 1):
                lower_bound = bins[i][j]
                upper_bound = bins[i][j + 1]
                in_range = (feature_values >= lower_bound) & (feature_values < upper_bound)
                if torch.any(in_range):
                    noise[in_range, i] = torch.randn(in_range.sum(), device=x.device) * torch.sigmoid(noise_params[i, j]) * self.noise_scale

        return x + noise


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.batch_norm = BatchNorm1d(num_numerical_types)

    def forward(self, x):
        x = self.batch_norm(x)
        return x


class NumEncoderTransformer(nn.Module):
    def __init__(self, num_continuous_features: int, embedding_dim: int, learninable_noise: bool, bins):
        super(NumEncoderTransformer, self).__init__()

        self.num_continuous_features = num_continuous_features
        self.kanclassifier = KAN([num_continuous_features, max(2 * num_continuous_features + 1, embedding_dim),
                                   embedding_dim * num_continuous_features])
        self.numerical_embedder = NumericalEmbedder(1, num_continuous_features)

        self.learninable_noise = learninable_noise
        self.bins = bins

        if learninable_noise:
            self.noise_module = NoiseLearnModule(num_continuous_features)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        if self.learninable_noise:
            x = self.noise_module(x, self.bins)

        x = self.numerical_embedder(x)
        x = x.squeeze(-1)
        x = self.kanclassifier(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str, dropout_rate: float):
        super(MLPBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm([output_dim]),
            Activation(activation),
            nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: str, dropout_rate: float):
        super(MLP, self).__init__()
        dims = [input_dim] + hidden_dims
        self.model = nn.Sequential(*(
            [MLPBlock(dims[i], dims[i + 1], activation, dropout_rate) for i in range(len(dims) - 1)] +
            [nn.Linear(dims[-1], output_dim)]))

    def forward(self, x):
        return self.model(x)


# ============== KAN и KANLinear ==============

class KAN(torch.nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                          scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                          base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)


class KANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1,
                 scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True,
                 base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) *
                self.curve2coeff(self.grid.T[self.spline_order: -self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
                                  self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat([grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                          grid,
                          grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1)], dim=0)
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy


# ============== KAN Transformer (опционально) ==============

class KANTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.kan_ffn = KAN([dim, dim * expansion_factor, dim])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)
        kan_out = self.kan_ffn(x_flat).view(batch, seq, dim)
        x = x + self.dropout(kan_out)
        x = self.norm2(x)
        return x, attn_weights


class KANTransformer(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            KANTransformerBlock(dim, num_heads, expansion_factor, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        all_attentions = []
        for block in self.blocks:
            x, attn = block(x)
            all_attentions.append(attn)
        return x, all_attentions


# ============== Стандартный Transformer ==============

class StandardTransformer(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm([dim]))

    def forward(self, x):
        # Для совместимости с KANTransformer возвращаем (x, None)
        return self.transformer(x), None


# ============== TabKANet с выбором типа трансформера ==============

class TabKANet(nn.Module):
    def __init__(self,
                 output_dim: int,
                 vocabulary: Dict[str, Dict[str, int]],
                 num_continuous_features: int,
                 embedding_dim: int,
                 nhead: int,
                 num_layers: int,
                 dim_feedforward: int,
                 attn_dropout_rate: float,
                 mlp_hidden_dims: List[int],
                 activation: str,
                 ffn_dropout_rate: float,
                 learninable_noise: bool,
                 bins: Optional[List[Tensor]],
                 classifier_type: str = 'mlp',
                 transformer_type: Literal['standard', 'kan'] = 'standard',
                 return_attention: bool = False):
        super(TabKANet, self).__init__()

        self.embedding_dim = embedding_dim
        self.len_vocabulary = len(vocabulary)
        self.num_continuous_features = num_continuous_features
        self.return_attention = return_attention
        self.transformer_type = transformer_type

        # Выбор типа трансформера
        if transformer_type == 'kan':
            self.transformer_model = KANTransformer(
                dim=embedding_dim,
                num_layers=num_layers,
                num_heads=nhead,
                expansion_factor=dim_feedforward // embedding_dim,
                dropout=attn_dropout_rate
            )
        else:  # standard
            self.transformer_model = StandardTransformer(
                dim=embedding_dim,
                num_layers=num_layers,
                num_heads=nhead,
                dim_feedforward=dim_feedforward,
                dropout=attn_dropout_rate
            )

        self.encoders = nn.ModuleDict({
            'categorical_feature_encoder': CatEncoder(vocabulary, embedding_dim),
            'continuous_feature_encoder': NumEncoderTransformer(num_continuous_features, embedding_dim,
                                                                  learninable_noise, bins),
        })

        if classifier_type == 'kan':
            self.classifier = KAN([embedding_dim * (len(vocabulary) + num_continuous_features)] + mlp_hidden_dims + [output_dim])
        else:
            self.classifier = MLP(embedding_dim * (len(vocabulary) + num_continuous_features), output_dim,
                                   mlp_hidden_dims, activation, ffn_dropout_rate)

    def forward(self, categorical_x: torch.Tensor, continuous_x: torch.Tensor):
        continuous_x = self.encoders['continuous_feature_encoder'](continuous_x)
        batch_size = continuous_x.size(0)

        if self.len_vocabulary == 0:
            x = continuous_x.view(batch_size, self.num_continuous_features, self.embedding_dim)
            x, attentions = self.transformer_model(x)
            x = x.view(batch_size, -1)
        else:
            continuous_x = continuous_x.view(batch_size, self.num_continuous_features, self.embedding_dim)
            x = self.encoders['categorical_feature_encoder'](categorical_x, continuous_x)
            x, attentions = self.transformer_model(x)
            x = x.view(batch_size, -1)

        logits = self.classifier(x)

        if self.return_attention and attentions is not None:
            return logits, attentions
        else:
            return logits