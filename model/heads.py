"""
Prediction heads for token classification and coordinate regression.
"""
import torch
import torch.nn as nn


class TokenClassificationHead(nn.Module):
    """
    Classification head for predicting special tokens.

    Predicts one of: <coord>, <sep>, <eos>
    """

    def __init__(self, hidden_dim=256, num_special_tokens=3):
        """
        Args:
            hidden_dim: input hidden dimension
            num_special_tokens: number of special token types
                - 0: <coord>
                - 1: <sep>
                - 2: <eos>
        """
        super().__init__()

        self.num_special_tokens = num_special_tokens

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_special_tokens)
        )

    def forward(self, x):
        """
        Predict token class.

        Args:
            x: [B, T, D] - decoder outputs

        Returns:
            logits: [B, T, num_special_tokens]
        """
        return self.head(x)


class CoordinateRegressionHead(nn.Module):
    """
    Regression head for predicting (x, y) coordinates.

    Outputs are passed through sigmoid to ensure [0, 1] range.
    """

    def __init__(self, hidden_dim=256):
        """
        Args:
            hidden_dim: input hidden dimension
        """
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()  # Ensure output in [0, 1]
        )

    def forward(self, x):
        """
        Predict coordinates.

        Args:
            x: [B, T, D] - decoder outputs

        Returns:
            coords: [B, T, 2] - predicted (x, y) in [0, 1]^2
        """
        return self.head(x)


class Raster2SeqHeads(nn.Module):
    """
    Combined prediction heads for Raster2Seq model.

    Contains both token classification and coordinate regression heads,
    and can apply them to intermediate decoder layer outputs for auxiliary losses.
    """

    def __init__(self, hidden_dim=256, num_special_tokens=3):
        """
        Args:
            hidden_dim: decoder hidden dimension
            num_special_tokens: number of special token types
        """
        super().__init__()

        self.token_head = TokenClassificationHead(hidden_dim, num_special_tokens)
        self.coord_head = CoordinateRegressionHead(hidden_dim)

    def forward(self, decoder_output):
        """
        Apply prediction heads.

        Args:
            decoder_output: [B, T, D] - decoder outputs

        Returns:
            dict with:
                - token_logits: [B, T, num_special_tokens]
                - coords: [B, T, 2]
        """
        token_logits = self.token_head(decoder_output)
        coords = self.coord_head(decoder_output)

        return {
            'token_logits': token_logits,
            'coords': coords
        }

    def forward_auxiliary(self, layer_outputs):
        """
        Apply prediction heads to intermediate layer outputs for auxiliary losses.

        Args:
            layer_outputs: list of [B, T, D] tensors from each decoder layer

        Returns:
            list of dicts, each containing:
                - token_logits: [B, T, num_special_tokens]
                - coords: [B, T, 2]
        """
        auxiliary_outputs = []

        for layer_out in layer_outputs:
            token_logits = self.token_head(layer_out)
            coords = self.coord_head(layer_out)

            auxiliary_outputs.append({
                'token_logits': token_logits,
                'coords': coords
            })

        return auxiliary_outputs


class SequenceEmbedding(nn.Module):
    """
    Embedding module for decoder input sequences.

    Handles:
    - Special token embeddings (<coord>, <sep>, <eos>)
    - Coordinate value embeddings
    """

    def __init__(self, hidden_dim=256, num_special_tokens=3):
        """
        Args:
            hidden_dim: embedding dimension
            num_special_tokens: number of special token types
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Special token embeddings
        self.special_token_embed = nn.Embedding(num_special_tokens, hidden_dim)

        # Coordinate value embedding
        self.coord_embed = nn.Linear(2, hidden_dim)

        # Learned embedding to distinguish token type (special vs coordinate)
        self.token_type_embed = nn.Embedding(2, hidden_dim)  # 0: special, 1: coordinate

    def embed_special_token(self, token_ids):
        """
        Embed special tokens.

        Args:
            token_ids: [B, T] - token IDs (0: <coord>, 1: <sep>, 2: <eos>)

        Returns:
            embeddings: [B, T, D]
        """
        token_embeds = self.special_token_embed(token_ids)
        type_embeds = self.token_type_embed(torch.zeros_like(token_ids))  # Type 0 for special tokens

        return token_embeds + type_embeds

    def embed_coordinates(self, coords):
        """
        Embed coordinate values.

        Args:
            coords: [B, T, 2] - (x, y) coordinates

        Returns:
            embeddings: [B, T, D]
        """
        B, T = coords.shape[:2]
        device = coords.device

        coord_embeds = self.coord_embed(coords)
        type_embeds = self.token_type_embed(torch.ones(B, T, device=device, dtype=torch.long))  # Type 1 for coords

        return coord_embeds + type_embeds

    def forward(self, sequence_data):
        """
        Embed a mixed sequence of special tokens and coordinates.

        This is a flexible interface that can handle different input formats.

        Args:
            sequence_data: dict with:
                - special_token_ids: [B, T] (optional)
                - coords: [B, T, 2] (optional)
                - mask: [B, T] indicating which positions are coords vs special tokens

        Returns:
            embeddings: [B, T, D]
        """
        raise NotImplementedError(
            "This method needs to be customized based on your sequence format. "
            "Use embed_special_token() and embed_coordinates() separately."
        )
