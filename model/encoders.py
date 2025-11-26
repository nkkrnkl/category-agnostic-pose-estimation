"""
Encoder modules for query images and support poses.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import math


class QueryImageEncoder(nn.Module):
    """
    ResNet-50 encoder for query images.
    Produces feature maps of shape [B, 2048, 16, 16] for 512x512 input.
    """

    def __init__(self, pretrained=True, frozen_stages=2):
        """
        Args:
            pretrained: whether to use ImageNet-pretrained weights
            frozen_stages: number of initial stages to freeze (0-4)
        """
        super().__init__()

        # Load pretrained ResNet-50
        # Use new weights API (torchvision 0.13+)
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        resnet = models.resnet50(weights=weights)

        # Remove FC layer and avgpool
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # /4
        self.layer2 = resnet.layer2  # /8
        self.layer3 = resnet.layer3  # /16
        self.layer4 = resnet.layer4  # /32

        # Freeze stages
        self._freeze_stages(frozen_stages)

        self.out_channels = 2048

    def _freeze_stages(self, frozen_stages):
        """Freeze initial stages of ResNet."""
        if frozen_stages >= 1:
            self.bn1.eval()
            for param in [self.conv1.parameters(), self.bn1.parameters()]:
                for p in param:
                    p.requires_grad = False

        for i in range(1, frozen_stages + 1):
            if i <= 4:
                layer = getattr(self, f'layer{i}')
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: [B, 3, 512, 512] - query images

        Returns:
            features: [B, 2048, 16, 16] - feature maps
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class SupportPoseEncoder(nn.Module):
    """
    Transformer encoder for support pose (keypoint coordinates).

    Embeds support keypoints with positional and graph-based encodings,
    then processes with a transformer encoder.
    """

    def __init__(
        self,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        max_keypoints=100
    ):
        """
        Args:
            hidden_dim: embedding dimension
            num_layers: number of transformer encoder layers
            num_heads: number of attention heads
            dropout: dropout rate
            max_keypoints: maximum number of keypoints
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_keypoints = max_keypoints

        # Coordinate embedding
        self.coord_embed = nn.Linear(2, hidden_dim)

        # Learned keypoint ID embedding
        self.keypoint_id_embed = nn.Embedding(max_keypoints, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, support_coords, keypoint_mask=None):
        """
        Encode support pose coordinates.

        Args:
            support_coords: [B, T_max, 2] - normalized coordinates in [0, 1]^2
            keypoint_mask: [B, T_max] - boolean mask (True for valid keypoints)

        Returns:
            support_embeddings: [B, T_max, hidden_dim]
        """
        B, T = support_coords.shape[:2]
        device = support_coords.device

        # Embed coordinates
        coord_embeds = self.coord_embed(support_coords)  # [B, T, D]

        # Add keypoint ID embeddings
        keypoint_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
        id_embeds = self.keypoint_id_embed(keypoint_ids)  # [B, T, D]

        # Combine embeddings
        embeddings = coord_embeds + id_embeds  # [B, T, D]

        # Create attention mask for transformer
        # For nn.TransformerEncoder, the mask should be:
        # - positions with True are NOT allowed to attend
        # - positions with False ARE allowed to attend
        if keypoint_mask is not None:
            # Invert the mask: keypoint_mask has True for valid, we need True for invalid
            src_key_padding_mask = ~keypoint_mask  # [B, T]
        else:
            src_key_padding_mask = None

        # Apply transformer encoder
        encoded = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )

        # Layer norm
        encoded = self.norm(encoded)

        return encoded


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for spatial feature maps.
    """

    def __init__(self, hidden_dim, max_size=64):
        """
        Args:
            hidden_dim: embedding dimension (must be divisible by 4)
            max_size: maximum spatial dimension
        """
        super().__init__()

        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4"

        self.hidden_dim = hidden_dim

        # Create positional encodings
        pe = torch.zeros(max_size, max_size, hidden_dim)

        # Create position indices
        y_pos = torch.arange(0, max_size).unsqueeze(1).expand(max_size, max_size)
        x_pos = torch.arange(0, max_size).unsqueeze(0).expand(max_size, max_size)

        # Compute frequency bands
        div_term = torch.exp(
            torch.arange(0, hidden_dim // 2, 2).float() * (-math.log(10000.0) / (hidden_dim // 2))
        )

        # Apply sin/cos to x and y positions
        pe[:, :, 0:hidden_dim // 4] = torch.sin(x_pos.unsqueeze(2) * div_term)
        pe[:, :, hidden_dim // 4:hidden_dim // 2] = torch.cos(x_pos.unsqueeze(2) * div_term)
        pe[:, :, hidden_dim // 2:3 * hidden_dim // 4] = torch.sin(y_pos.unsqueeze(2) * div_term)
        pe[:, :, 3 * hidden_dim // 4:hidden_dim] = torch.cos(y_pos.unsqueeze(2) * div_term)

        # Register as buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add 2D positional encoding to feature maps.

        Args:
            x: [B, C, H, W] - feature maps

        Returns:
            x_with_pe: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Get positional encoding
        pe = self.pe[:H, :W, :C]  # [H, W, C]
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        return x + pe
