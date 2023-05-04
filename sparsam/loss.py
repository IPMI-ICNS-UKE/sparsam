from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseDinoLoss(nn.Module, ABC):
    def __init__(
            self,
            n_crops: int = 7,
            student_temp: float = 0.1,
            teacher_temp: float = 0.04,

            warmup_teacher_temp: float = 0.04,
            warmup_teacher_temp_iterations: int = 0,
    ):
        super().__init__()
        self.n_crops = n_crops
        self.step = 0
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_iterations = warmup_teacher_temp_iterations
        if not warmup_teacher_temp_iterations or warmup_teacher_temp_iterations == 0:
            self.teacher_temp_slope = 0
        else:
            self.teacher_temp_slope = (teacher_temp - warmup_teacher_temp) / warmup_teacher_temp_iterations

    @abstractmethod
    def forward(self, student_output: Tensor, teacher_output: Tensor, step: int = None) -> Tensor:
        pass

    def _teacher_temp_schedule(self, step: int) -> float:
        if step > self.warmup_teacher_temp_iterations:
            teacher_temp = self.teacher_temp
        else:
            teacher_temp = self.warmup_teacher_temp + self.teacher_temp_slope * step
        return teacher_temp


class DINOLoss(BaseDinoLoss):
    # Copyright (c) Facebook, Inc. and its affiliates.
    def __init__(
            self,
            n_crops: int = 7,
            student_temp: float = 0.1,
            warmup_teacher_temp: float = 0.04,
            teacher_temp: float = 0.04,
            warmup_teacher_temp_iterations: int = 0,
            center_momentum: float = 0.9,
            out_dim=None,
    ):
        super().__init__(
            n_crops=n_crops,
            student_temp=student_temp,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_iterations=warmup_teacher_temp_iterations,
        )
        self.center_momentum = center_momentum
        if not out_dim:
            center = torch.empty(0)
        else:
            center = torch.zeros(1, out_dim)
        self.register_buffer("center", center, persistent=True)

    def forward(self, student_output: Tensor, teacher_output: Tensor, step: int = None) -> Tensor:
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
        teacher_out = self._prepare_teacher_output(teacher_output)
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

        self._update_center(teacher_output)
        self.step += 1
        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_output: Tensor):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        if self.center.numel() == 0 or (self.center == 0).all():
            self.center = batch_center
        else:
            # ema update
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def _prepare_teacher_output(self, teacher_output: Tensor) -> Tensor:
        temp = self._teacher_temp_schedule(self.step)
        if self.center.numel() == 0 or (self.center == 0).all():
            center = torch.mean(teacher_output, dim=0, keepdim=True)
        else:
            center = self.center
        teacher_out = (teacher_output - center) / temp
        teacher_out = F.softmax(teacher_out, dim=-1)
        return teacher_out
