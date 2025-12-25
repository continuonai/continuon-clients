from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """Loss for Teacher-Student training (Knowledge Distillation)."""

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate distillation loss.
        - student_logits: [B, V]
        - teacher_logits: [B, V]
        - labels: [B]
        """
        # Soft targets from teacher
        soft_targets = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distillation_loss = self.kl_div(soft_targets, soft_teacher) * (self.temperature ** 2)
        
        # Standard cross entropy with hard labels
        student_loss = self.ce_loss(student_logits, labels)
        
        return self.alpha * student_loss + (1 - self.alpha) * distillation_loss
