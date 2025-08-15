import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import librosa

# =============================================
# MODULAR MODEL COMPONENTS
# =============================================

class AdaptiveCNN(nn.Module):
    """Adaptive CNN for audio feature extraction(AudioFeatureExtractor)"""
    
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(dropout)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout2d(dropout)
        
        self.conv4 = nn.Conv2d(256, d_model, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(d_model)
    
    def adaptive_conv_block(self, x, conv_layer, bn_layer, dropout_layer, apply_pooling=True):
        """Apply convolution with adaptive padding and optional pooling"""
        kernel_h, kernel_w = conv_layer.kernel_size
        pad_h = (kernel_h - 1) // 2
        pad_w = (kernel_w - 1) // 2
        
        input_h, input_w = x.shape[2], x.shape[3]
        pad_h = min(pad_h, input_h // 2)
        pad_w = min(pad_w, input_w // 2)
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        
        x = conv_layer(x)
        x = bn_layer(x)
        x = F.gelu(x)
        
        if apply_pooling and x.shape[2] >= 2 and x.shape[3] >= 2:
            target_h = max(1, x.shape[2] // 2)
            target_w = max(1, x.shape[3] // 2)
            x = F.adaptive_avg_pool2d(x, (target_h, target_w))
        
        x = dropout_layer(x)
        return x
    
    def forward(self, x):
        batch_size, _, n_mels, time_steps = x.shape
        
        min_height, min_width = 8, 8
        if n_mels < min_height or time_steps < min_width:
            pad_h = max(0, min_height - n_mels)
            pad_w = max(0, min_width - time_steps)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        x = self.adaptive_conv_block(x, self.conv1, self.bn1, self.dropout1, apply_pooling=True)
        x = self.adaptive_conv_block(x, self.conv2, self.bn2, self.dropout2, apply_pooling=True)
        x = self.adaptive_conv_block(x, self.conv3, self.bn3, self.dropout3, apply_pooling=True)
        x = self.adaptive_conv_block(x, self.conv4, self.bn4, nn.Identity(), apply_pooling=False)
        
        target_time_dim = min(x.shape[3], min_height*min_width)
        x = F.adaptive_avg_pool2d(x, (1, target_time_dim))
        
        return x

class ProjectionHead(nn.Module):
    """Projection head for embeddings"""
    
    def __init__(self, input_dim, projection_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

class ModalityClassifier(nn.Module):
    """Modality classifier for domain adaptation"""
    
    def __init__(self, input_dim, inner_dim=256, num_classes=2, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class AudioEncoder(nn.Module):
    """Audio encoder with CNN + Transformer"""
    
    def __init__(self, n_mels, embed_dim, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.cnn = AdaptiveCNN(d_model, dropout) # AudioFeatureExtractor
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)
        self.projection = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, mel_spec):
        x = mel_spec.unsqueeze(1)  # Add channel dimension
        x = self.cnn(x)  # CNN feature extraction
        x = x.squeeze(2)  # Remove frequency dimension
        x = x.transpose(1, 2)  # (batch, time, features)
        x = self.transformer(x)  # Transformer encoding
        x = x.mean(dim=1)  # Global average pooling
        x = self.projection(x)  # Final projection
        return x

class TextEncoder(nn.Module):
    """Text encoder using XLM-RoBERTa"""
    
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('xlm-roberta-base')
        
        # Freeze some layers for efficiency
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use mean pooling instead of just CLS token
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

# =============================================
# MAIN CLAMP MODEL
# =============================================

class CLAMPModel(nn.Module):
    """CLAMP model with multilingual support and variable-length audio"""
    
    def __init__(self, 
                 audio_embed_dim=512,
                 text_embed_dim=768,
                 projection_dim=512,
                 class_inner_dim=256,
                 d_model=512, nhead=8, dim_feedforward=2048,
                 n_mels=80,
                 temperature=0.07,
                 use_multihead_attention=True,
                 dropout=0.1):
        
        super().__init__()
        
        self.audio_embed_dim = audio_embed_dim
        self.text_embed_dim = text_embed_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Modular components
        self.audio_encoder = AudioEncoder(n_mels, audio_embed_dim, d_model, nhead, dim_feedforward, dropout)
        self.text_encoder = TextEncoder()
        
        self.audio_projection_head = ProjectionHead(audio_embed_dim, projection_dim, dropout)
        self.text_projection_head = ProjectionHead(text_embed_dim, projection_dim, dropout)
        
        # Cross-modal attention
        if use_multihead_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=projection_dim,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.cross_attention = None
        
        # Modality classifiers
        self.audio_modality_classifier = ModalityClassifier(projection_dim, class_inner_dim, 2, dropout)
        self.text_modality_classifier = ModalityClassifier(projection_dim, class_inner_dim, 2, dropout)
        
        # Sigmoid loss parameters
        self.sigmoid_a = nn.Parameter(torch.tensor(10.0))
        self.sigmoid_b = nn.Parameter(torch.tensor(-10.0))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def encode_audio(self, mel_spec):
        """Encode mel spectrogram to audio embeddings"""
        return self.audio_encoder(mel_spec)
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text to text embeddings"""
        return self.text_encoder(input_ids, attention_mask)
    
    def forward(self, audio_features, text_input_ids, text_attention_mask, return_cross_attention=False, return_modality_logits=False):
        """Forward pass"""
        # Encode modalities
        audio_embeddings = self.encode_audio(audio_features)
        text_embeddings = self.encode_text(text_input_ids, text_attention_mask)
        
        # Project to joint space
        audio_proj = self.audio_projection_head(audio_embeddings)
        text_proj = self.text_projection_head(text_embeddings)
        
        # L2 normalize projections
        audio_proj_norm = F.normalize(audio_proj, p=2, dim=-1)
        text_proj_norm = F.normalize(text_proj, p=2, dim=-1)
        
        outputs = {
            'audio_embeddings': audio_embeddings,
            'text_embeddings': text_embeddings,
            'audio_proj': audio_proj_norm,
            'text_proj': text_proj_norm,
            'sigmoid_a': self.sigmoid_a,
            'sigmoid_b': self.sigmoid_b
        }
        
        # Cross-modal attention
        if return_cross_attention and self.cross_attention is not None:
            audio_attended, audio_attn_weights = self.cross_attention(
                audio_proj.unsqueeze(1),
                text_proj.unsqueeze(1),
                text_proj.unsqueeze(1)
            )
            
            text_attended, text_attn_weights = self.cross_attention(
                text_proj.unsqueeze(1),
                audio_proj.unsqueeze(1),
                audio_proj.unsqueeze(1)
            )
            
            outputs.update({
                'audio_attended': audio_attended.squeeze(1),
                'text_attended': text_attended.squeeze(1),
                'audio_attn_weights': audio_attn_weights,
                'text_attn_weights': text_attn_weights
            })
        
        # Modality classification
        if return_modality_logits:
            audio_modality_logits = self.audio_modality_classifier(audio_proj)
            text_modality_logits = self.text_modality_classifier(text_proj)
            
            outputs.update({
                'audio_modality_logits': audio_modality_logits,
                'text_modality_logits': text_modality_logits
            })
        
        return outputs