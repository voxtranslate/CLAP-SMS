import torch
import torch.nn as nn


# =============================================
# LOSS FUNCTIONS
# =============================================

class CLAMPLoss(nn.Module):
    """Loss function combining original CLAMP losses with sigmoid loss"""
    
    def __init__(self, 
                 temperature=0.07,
                 contrastive_weight=1.0,
                 modality_weight=0.1,
                 cross_attention_weight=0.05,
                 sigmoid_weight=1.0,
                 label_smoothing=0.1):
        super().__init__()
        
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.modality_weight = modality_weight
        self.cross_attention_weight = cross_attention_weight
        self.sigmoid_weight = sigmoid_weight
        self.label_smoothing = label_smoothing
        
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.mse_loss = nn.MSELoss()
    
    def contrastive_loss(self, audio_proj, text_proj, return_logits=False):
        """Contrastive loss (InfoNCE)"""
        batch_size = audio_proj.shape[0]
        
        # Clean NaN and Inf values
        audio_proj = torch.nan_to_num(audio_proj, nan=0.0, posinf=1.0, neginf=-1.0)
        text_proj = torch.nan_to_num(text_proj, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize projections
        audio_proj = F.normalize(audio_proj, p=2, dim=-1)
        text_proj = F.normalize(text_proj, p=2, dim=-1)
        
        # Compute similarity matrix
        temperature = max(0.01, min(1.0, self.temperature))  # Clamp temperature
        similarity_matrix = torch.matmul(audio_proj, text_proj.T) / temperature
        
        # Clamp similarities to prevent overflow
        similarity_matrix = torch.clamp(similarity_matrix, min=-20.0, max=20.0)
        
        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(batch_size, device=audio_proj.device, dtype=torch.long)
        
        # Compute both directions of contrastive loss
        loss_audio_to_text = F.cross_entropy(similarity_matrix, labels, label_smoothing=self.label_smoothing)
        loss_text_to_audio = F.cross_entropy(similarity_matrix.T, labels, label_smoothing=self.label_smoothing)
        
        total_loss = (loss_audio_to_text + loss_text_to_audio) / 2.0
        
        if return_logits:
            return total_loss, similarity_matrix
        return total_loss
    
    def sigmoid_loss(self, audio_proj, text_proj, sigmoid_a, sigmoid_b):
        """Sigmoid loss with learnable parameters"""
        batch_size = audio_proj.shape[0]
        
        similarities = torch.matmul(audio_proj, text_proj.T)
        
        pos_mask = torch.eye(batch_size, device=audio_proj.device)
        neg_mask = 1 - pos_mask
        
        pos_similarities = similarities * pos_mask
        neg_similarities = similarities * neg_mask
        
        pos_loss = -torch.log(torch.sigmoid(sigmoid_a * pos_similarities + sigmoid_b) + 1e-8)
        neg_loss = -torch.log(1 - torch.sigmoid(sigmoid_a * neg_similarities + sigmoid_b) + 1e-8)
        
        pos_loss = (pos_loss * pos_mask).sum() / pos_mask.sum()
        neg_loss = (neg_loss * neg_mask).sum() / neg_mask.sum()
        
        return pos_loss + neg_loss
    
    def modality_classification_loss(self, modality_logits, modality_labels):
        """Modality classification loss for domain adaptation"""
        return self.cross_entropy(modality_logits, modality_labels)
    
    def cross_attention_alignment_loss(self, audio_attended, text_attended, audio_proj, text_proj):
        """Cross-attention alignment loss"""
        audio_alignment_loss = self.mse_loss(audio_attended, text_proj)
        text_alignment_loss = self.mse_loss(text_attended, audio_proj)
        return (audio_alignment_loss + text_alignment_loss) / 2
    
    def forward(self, outputs, modality_labels=None):
        """Compute total loss combining all components"""
        losses = {}
        device = outputs['audio_proj'].device
        
        # Contrastive loss
        contrastive_loss, similarity_logits = self.contrastive_loss(
            outputs['audio_proj'], 
            outputs['text_proj'],
            return_logits=True
        )
        losses['contrastive'] = contrastive_loss
        
        # Sigmoid loss
        sigmoid_loss = self.sigmoid_loss(
            outputs['audio_proj'],
            outputs['text_proj'], 
            outputs['sigmoid_a'],
            outputs['sigmoid_b']
        )
        losses['sigmoid'] = sigmoid_loss
        
        # Modality classification loss
        if modality_labels is not None and 'audio_modality_logits' in outputs:
            audio_modality_loss = self.modality_classification_loss(
                outputs['audio_modality_logits'], 
                modality_labels
            )
            text_modality_loss = self.modality_classification_loss(
                outputs['text_modality_logits'], 
                modality_labels
            )
            modality_loss = (audio_modality_loss + text_modality_loss) / 2
            losses['modality'] = modality_loss
        else:
            modality_loss = torch.tensor(0.0, device=device)
            losses['modality'] = modality_loss
        
        # Cross-attention alignment loss
        if 'audio_attended' in outputs and 'text_attended' in outputs:
            alignment_loss = self.cross_attention_alignment_loss(
                outputs['audio_attended'],
                outputs['text_attended'],
                outputs['audio_proj'],
                outputs['text_proj']
            )
            losses['alignment'] = alignment_loss
        else:
            alignment_loss = torch.tensor(0.0, device=device)
            losses['alignment'] = alignment_loss
        
        # Total loss
        total_loss = (
            self.contrastive_weight * contrastive_loss +
            self.sigmoid_weight * sigmoid_loss +
            self.modality_weight * modality_loss +
            self.cross_attention_weight * alignment_loss
        )
        
        # Additional metrics
        with torch.no_grad():
            if similarity_logits.shape[0] > 0 and similarity_logits.shape[1] > 0:
                predictions = similarity_logits.argmax(dim=1)
                labels = torch.arange(len(predictions), device=predictions.device)
                accuracy = (predictions == labels).float().mean()
                losses['accuracy'] = accuracy
                
                positive_similarities = torch.diag(similarity_logits)
                losses['avg_similarity'] = positive_similarities.mean()
            else:
                losses['accuracy'] = torch.tensor(0.0, device=device)
                losses['avg_similarity'] = torch.tensor(0.0, device=device)
            
            losses['sigmoid_a'] = outputs['sigmoid_a'].item()
            losses['sigmoid_b'] = outputs['sigmoid_b'].item()
        
        return total_loss, losses