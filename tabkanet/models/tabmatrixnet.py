from typing import Dict, List, Optional
import torch
import torch.nn as nn
from .base_blocks import ColumnEmbedding

# Импорт твоего MatrixKAN
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .MatrixKANLayer import MatrixKANLayer
from .MatrixKAN import MatrixKAN


class CatEncoder(nn.Module):
    def __init__(self, vocabulary: Dict[str, Dict[str, int]], embedding_dim: int):
        super(CatEncoder, self).__init__()
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.columnembedding = ColumnEmbedding(vocabulary, embedding_dim)

    def forward(self, x: torch.Tensor, continuous_x_res: torch.Tensor):
        x = [self.columnembedding(x[:, i], col) for i, col in enumerate(self.vocabulary)]
        x = torch.stack(x, dim=1)
        x = torch.cat((x, continuous_x_res), dim=1)
        return x


class NumEncoderMatrix(nn.Module):
    def __init__(self, num_continuous_features: int, embedding_dim: int, learninable_noise: bool, bins):
        super(NumEncoderMatrix, self).__init__()
        
        self.num_continuous_features = num_continuous_features
        
        hidden_dim = max(2 * num_continuous_features + 1, embedding_dim)
        self.matrix_kan = MatrixKAN(
            width=[num_continuous_features, hidden_dim, embedding_dim * num_continuous_features],
            grid=5,
            k=3,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.learninable_noise = learninable_noise
        self.bins = bins

    def forward(self, x: torch.Tensor):
        x = self.matrix_kan(x)
        return x


class TabMatrixNet(nn.Module):
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
                 learninable_noise: bool = False, 
                 bins: Optional[List[torch.Tensor]] = None):
        super(TabMatrixNet, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.len_vocabulary = len(vocabulary)
        self.num_continuous_features = num_continuous_features
        
        # Transformer encoder
        self.transformer_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=attn_dropout_rate,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm([embedding_dim])
        )
        
        # Encoders
        self.encoders = nn.ModuleDict({
            'categorical_feature_encoder': CatEncoder(vocabulary, embedding_dim),
            'continuous_feature_encoder': NumEncoderMatrix(
                num_continuous_features, embedding_dim, learninable_noise, bins
            ),
        })
        
        # MLP Classifier
        from .basic_mlp import MLP
        self.classifier = MLP(
            embedding_dim * (len(vocabulary) + num_continuous_features),
            output_dim,
            mlp_hidden_dims,
            activation,
            ffn_dropout_rate
        )
    
    def forward(self, categorical_x: torch.Tensor, continuous_x: torch.Tensor):
        continuous_x = self.encoders['continuous_feature_encoder'](continuous_x)
        batch_size = continuous_x.size(0)

        if self.len_vocabulary == 0:
            x = continuous_x.view(batch_size, self.num_continuous_features, self.embedding_dim)
            x = self.transformer_model(x).view(batch_size, -1)
            x = self.classifier(x)
        else:
            continuous_x = continuous_x.view(batch_size, self.num_continuous_features, self.embedding_dim)
            x = self.encoders['categorical_feature_encoder'](categorical_x, continuous_x)
            x_transformed = self.transformer_model(x)
            x_transformed_flat = x_transformed.view(batch_size, -1)
            x = self.classifier(x_transformed_flat)
        
        return x