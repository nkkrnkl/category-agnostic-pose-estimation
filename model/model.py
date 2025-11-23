"""
Raster2Seq model for Category-Agnostic Pose Estimation.

Integrates query image encoder, support pose encoder, autoregressive decoder,
and prediction heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoders import QueryImageEncoder, SupportPoseEncoder
from model.decoder import AutoregressiveDecoder
from model.heads import Raster2SeqHeads, SequenceEmbedding


# Special token IDs
TOKEN_COORD = 0
TOKEN_SEP = 1
TOKEN_EOS = 2


class Raster2SeqCAPE(nn.Module):
    """
    Raster2Seq model for category-agnostic pose estimation.

    Architecture:
    1. Query Image Encoder (ResNet-50)
    2. Support Pose Encoder (Transformer)
    3. Autoregressive Decoder (Transformer with cross-attention)
    4. Prediction Heads (token classification + coordinate regression)
    """

    def __init__(
        self,
        hidden_dim=256,
        num_encoder_layers=3,
        num_decoder_layers=6,
        num_heads=8,
        dropout=0.1,
        max_keypoints=100,
        pretrained_resnet=True
    ):
        """
        Args:
            hidden_dim: model hidden dimension
            num_encoder_layers: number of support encoder layers
            num_decoder_layers: number of decoder layers
            num_heads: number of attention heads
            dropout: dropout rate
            max_keypoints: maximum number of keypoints
            pretrained_resnet: whether to use ImageNet-pretrained ResNet
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_keypoints = max_keypoints

        # Query image encoder
        self.query_encoder = QueryImageEncoder(pretrained=pretrained_resnet)

        # Support pose encoder
        self.support_encoder = SupportPoseEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_keypoints=max_keypoints
        )

        # Sequence embedding for decoder inputs
        self.sequence_embedding = SequenceEmbedding(
            hidden_dim=hidden_dim,
            num_special_tokens=3
        )

        # Autoregressive decoder
        self.decoder = AutoregressiveDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            image_feature_dim=2048,
            max_seq_length=max_keypoints * 4 + 10
        )

        # Prediction heads
        self.heads = Raster2SeqHeads(
            hidden_dim=hidden_dim,
            num_special_tokens=3
        )

    def forward(
        self,
        query_images,
        support_coords,
        target_coords=None,
        num_keypoints=None,
        keypoint_mask=None,
        teacher_forcing=True
    ):
        """
        Forward pass with teacher forcing (training) or autoregressive generation (inference).

        Args:
            query_images: [B, 3, 512, 512] - query images
            support_coords: [B, T_max, 2] - support coordinates (normalized)
            target_coords: [B, T_max, 2] - target coordinates (for teacher forcing)
            num_keypoints: [B] - actual number of keypoints per sample
            keypoint_mask: [B, T_max] - mask for valid keypoints
            teacher_forcing: whether to use teacher forcing

        Returns:
            dict with:
                - token_logits: [B, seq_len, 3]
                - coords: [B, seq_len, 2]
                - auxiliary_outputs: list of intermediate predictions
        """
        B, T_max = support_coords.shape[:2]
        device = query_images.device

        # Encode query image
        image_features = self.query_encoder(query_images)  # [B, 2048, H, W]

        # Encode support pose
        support_embeddings = self.support_encoder(
            support_coords,
            keypoint_mask=keypoint_mask
        )  # [B, T_max, D]

        if teacher_forcing and target_coords is not None:
            # Training mode: use teacher forcing
            outputs = self._forward_teacher_forcing(
                image_features=image_features,
                support_embeddings=support_embeddings,
                target_coords=target_coords,
                num_keypoints=num_keypoints,
                keypoint_mask=keypoint_mask
            )
        else:
            # Inference mode: autoregressive generation
            outputs = self._forward_autoregressive(
                image_features=image_features,
                support_embeddings=support_embeddings,
                num_keypoints=num_keypoints,
                keypoint_mask=keypoint_mask
            )

        return outputs

    def _forward_teacher_forcing(
        self,
        image_features,
        support_embeddings,
        target_coords,
        num_keypoints,
        keypoint_mask
    ):
        """
        Forward pass with teacher forcing for training.

        Builds the target sequence from ground truth:
        <coord>, x_1, y_1, <sep>, <coord>, x_2, y_2, <sep>, ..., <eos>

        Args:
            image_features: [B, 2048, H, W]
            support_embeddings: [B, T_max, D]
            target_coords: [B, T_max, 2]
            num_keypoints: [B]
            keypoint_mask: [B, T_max]

        Returns:
            dict with predictions
        """
        B, T_max = target_coords.shape[:2]
        device = target_coords.device

        # Build decoder input sequence
        # Format: <coord>, x_1, y_1, <sep>, <coord>, x_2, y_2, <sep>, ..., <eos>
        # Each keypoint generates 4 tokens: <coord>, x, y, <sep>
        # Plus 1 <eos> token at the end

        max_seq_len = T_max * 4 + 1  # 4 tokens per keypoint + 1 eos

        decoder_inputs = []

        for b in range(B):
            N = num_keypoints[b].item() if num_keypoints is not None else T_max

            seq_embeds = []

            for i in range(N):
                # <coord> token
                coord_token = self.sequence_embedding.embed_special_token(
                    torch.tensor([[TOKEN_COORD]], device=device, dtype=torch.long)
                ).squeeze(0)  # [1, D]

                # Coordinate embeddings
                coord_embed = self.sequence_embedding.embed_coordinates(
                    target_coords[b:b+1, i:i+1]  # [1, 1, 2]
                ).squeeze(0)  # [1, D]

                # <sep> token
                sep_token = self.sequence_embedding.embed_special_token(
                    torch.tensor([[TOKEN_SEP]], device=device, dtype=torch.long)
                ).squeeze(0)  # [1, D]

                seq_embeds.extend([coord_token, coord_embed, sep_token])

            # Add <eos> token
            eos_token = self.sequence_embedding.embed_special_token(
                torch.tensor([[TOKEN_EOS]], device=device, dtype=torch.long)
            ).squeeze(0)  # [1, D]
            seq_embeds.append(eos_token)

            # Stack into sequence
            seq = torch.cat(seq_embeds, dim=0)  # [seq_len, D]

            # Pad to max_seq_len if needed
            if seq.shape[0] < max_seq_len:
                padding = torch.zeros(
                    max_seq_len - seq.shape[0],
                    self.hidden_dim,
                    device=device
                )
                seq = torch.cat([seq, padding], dim=0)

            decoder_inputs.append(seq)

        decoder_inputs = torch.stack(decoder_inputs, dim=0)  # [B, max_seq_len, D]

        # Pass through decoder
        decoder_output, layer_outputs = self.decoder(
            tgt=decoder_inputs,
            image_features=image_features,
            support_embeddings=support_embeddings,
            support_mask=keypoint_mask
        )

        # Apply prediction heads
        predictions = self.heads(decoder_output)

        # Apply heads to intermediate layers for auxiliary losses
        auxiliary_outputs = self.heads.forward_auxiliary(layer_outputs)

        predictions['auxiliary_outputs'] = auxiliary_outputs

        return predictions

    def _forward_autoregressive(
        self,
        image_features,
        support_embeddings,
        num_keypoints,
        keypoint_mask,
        max_steps=500
    ):
        """
        Forward pass with autoregressive generation for inference.

        Args:
            image_features: [B, 2048, H, W]
            support_embeddings: [B, T_max, D]
            num_keypoints: [B] - expected number of keypoints
            keypoint_mask: [B, T_max]
            max_steps: maximum number of generation steps

        Returns:
            dict with:
                - predicted_coords: [B, N, 2]
                - token_sequence: list of generated tokens
        """
        B = image_features.shape[0]
        device = image_features.device

        # Initialize with <coord> token
        current_tokens = [
            self.sequence_embedding.embed_special_token(
                torch.tensor([[TOKEN_COORD]], device=device, dtype=torch.long)
            ) for _ in range(B)
        ]  # List of [B, 1, D]

        predicted_coords_list = [[] for _ in range(B)]
        finished = [False] * B

        for step in range(max_steps):
            # Stack current sequence
            current_seq = torch.cat(current_tokens, dim=1)  # [B, seq_len, D]

            # Decode
            decoder_output, _ = self.decoder(
                tgt=current_seq,
                image_features=image_features,
                support_embeddings=support_embeddings,
                support_mask=keypoint_mask
            )

            # Get predictions for last position
            last_output = decoder_output[:, -1:, :]  # [B, 1, D]

            # Predict token type and coordinates
            predictions = self.heads(last_output)
            token_logits = predictions['token_logits']  # [B, 1, 3]
            coords = predictions['coords']  # [B, 1, 2]

            # Greedy decoding: select most likely token
            predicted_tokens = token_logits.argmax(dim=-1)  # [B, 1]

            # Process each sample in batch
            for b in range(B):
                if finished[b]:
                    continue

                token = predicted_tokens[b, 0].item()

                if token == TOKEN_EOS:
                    finished[b] = True
                elif token == TOKEN_COORD:
                    # Next, we expect a coordinate
                    # For simplicity, we immediately add the predicted coordinate
                    predicted_coords_list[b].append(coords[b, 0].detach())

            # Check if all finished
            if all(finished):
                break

            # Prepare next input
            # Alternate between tokens and coordinates in a simplified manner
            # For this implementation, we follow: <coord> -> coord_value -> <sep> -> <coord> -> ...

            if step % 3 == 0:
                # After <coord>, embed the predicted coordinate
                next_embed = self.sequence_embedding.embed_coordinates(coords)
            elif step % 3 == 1:
                # After coordinate, add <sep>
                next_embed = self.sequence_embedding.embed_special_token(
                    torch.full((B, 1), TOKEN_SEP, device=device, dtype=torch.long)
                )
            else:
                # After <sep>, add <coord> or <eos>
                next_embed = self.sequence_embedding.embed_special_token(
                    predicted_tokens
                )

            current_tokens.append(next_embed)

        # Collect predicted coordinates
        max_coords = max(len(coords) for coords in predicted_coords_list)
        predicted_coords_batch = torch.zeros(B, max_coords, 2, device=device)

        for b in range(B):
            coords_b = torch.stack(predicted_coords_list[b]) if predicted_coords_list[b] else torch.zeros(0, 2, device=device)
            predicted_coords_batch[b, :len(coords_b)] = coords_b

        return {
            'predicted_coords': predicted_coords_batch
        }

    def compute_loss(
        self,
        outputs,
        target_coords,
        visibility,
        num_keypoints,
        coord_loss_weight=5.0
    ):
        """
        Compute training loss.

        Args:
            outputs: dict from forward pass
            target_coords: [B, T_max, 2]
            visibility: [B, T_max]
            num_keypoints: [B]
            coord_loss_weight: weight for coordinate loss

        Returns:
            dict with loss components
        """
        B, T_max = target_coords.shape[:2]
        device = target_coords.device

        # Get predictions
        # pred_coords shape: [B, seq_len, 2] where seq_len = 3*N + 1 (for N keypoints)
        pred_coords = outputs['coords']

        # Extract coordinate predictions at the right positions
        # In the sequence: <coord>, coord_value, <sep>, <coord>, coord_value, <sep>, ..., <eos>
        # Coordinate predictions are at positions 1, 4, 7, 10, ... (i.e., 1 + 3*i)

        coord_losses = []

        for b in range(B):
            N = num_keypoints[b].item()

            for i in range(N):
                # Position in sequence where this coordinate appears
                pos = 1 + 3 * i

                if pos < pred_coords.shape[1]:
                    # Predicted coordinate at this position
                    pred_coord = pred_coords[b, pos]  # [2]

                    # Target coordinate
                    target_coord = target_coords[b, i]  # [2]

                    # Visibility
                    vis = visibility[b, i].item()

                    if vis > 0:
                        # Compute L1 loss
                        loss = F.l1_loss(pred_coord, target_coord, reduction='mean')
                        coord_losses.append(loss)

        # Average coordinate loss
        if len(coord_losses) > 0:
            coord_loss = torch.stack(coord_losses).mean()
        else:
            coord_loss = torch.tensor(0.0, device=device)

        # Token classification loss (not implemented for simplicity)
        token_loss = torch.tensor(0.0, device=device)

        total_loss = token_loss + coord_loss_weight * coord_loss

        return {
            'total_loss': total_loss,
            'token_loss': token_loss,
            'coord_loss': coord_loss
        }
