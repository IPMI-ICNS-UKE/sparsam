import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DINOLoss(nn.Module):
    # Adopted from Facebook, Inc. and its affiliates https://github.com/facebookresearch/dino
    def __init__(
        self,
        n_crops: int = 7,
        student_temp: float = 0.1,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_iterations: int = 0,
        center_momentum: float = 0.9,
    ) -> Tensor:
        super().__init__()
        self.n_crops = n_crops
        self.center_momentum = center_momentum
        self.center = 0
        self.step = 0
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_iterations = warmup_teacher_temp_iterations
        if not warmup_teacher_temp_iterations or warmup_teacher_temp_iterations == 0:
            self.teacher_temp_slope = 0
        else:
            self.teacher_temp_slope = (teacher_temp - warmup_teacher_temp) / warmup_teacher_temp_iterations

    def forward(self, student_output, teacher_output, step=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        if step:
            self.step = step
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.n_crops)
        chunk_size = student_out[0].shape[0]
        teacher_chunks = teacher_output.shape[0] // chunk_size

        # teacher centering and sharpening
        temp = self._teacher_temp_schedule(self.step)
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(teacher_chunks)
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        self.step += 1
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        if isinstance(self.center, int):
            self.center = batch_center
        else:
            # ema update
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def _teacher_temp_schedule(self, step):
        if step > self.warmup_teacher_temp_iterations:
            teacher_temp = self.teacher_temp
        else:
            teacher_temp = self.warmup_teacher_temp + self.teacher_temp_slope * step
        return teacher_temp
