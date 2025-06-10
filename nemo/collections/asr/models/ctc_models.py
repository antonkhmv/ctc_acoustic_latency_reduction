# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os
from math import ceil, floor
from typing import Any, Dict, List, Optional, Union, Iterable

import numpy as np
import random
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRModuleMixin, ASRTranscriptionMixin, InterCTCMixin, TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import GenericTranscriptionType, TranscriptionReturnType
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.parts.preprocessing.parsers import make_parser
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging
from numba import jit


__all__ = ['EncDecCTCModel']





@jit(nopython=True)
def numba_calculate_expected_delays(log_probs, input_lengths, blank_id):
    """
    Compute expected first and last blank occurrence time using Numba.
    """
    B, _, _ = log_probs.shape
    head_latency = np.zeros(B, dtype=np.float32)
    tail_latency = np.zeros(B, dtype=np.float32)
    
    for b in range(B):
        length = input_lengths[b]
        log_blank = log_probs[b, :, blank_id]
        log_non_blank = np.log(1 - np.exp(log_blank))
        log_blank_sum = 0.0
    
        for t in range(length):
            log_prob_first_blank = log_non_blank[t] + log_blank_sum
            head_latency[b] += t * np.exp(log_prob_first_blank)
            log_blank_sum += log_blank[t]

        log_blank_sum = 0.0
        for t in range(length - 1, -1, -1):
            log_prob_first_blank = log_non_blank[t] + log_blank_sum
            tail_latency[b] += t * np.exp(log_prob_first_blank)
            log_blank_sum += log_blank[t]

    return head_latency, tail_latency


class TrimTail:
    def __init__(
        self,
        t_min_sec: float = 0.010,
        t_max_sec: float = 1.000,
        frame_size: float = 0.010,
    ):
        self.t_min = round(t_min_sec / frame_size)
        self.t_max = round(t_max_sec / frame_size)

    def __call__(self, processed_signal, processed_signal_length):
        """
        Args:
            processed_signal: Tensor (B, D, T)
            processed_signal_length: Tensor (B,)
        Returns:
            Trimmed processed_signal and updated processed_signal_length
        """
        B, _, T = processed_signal.shape
        device = processed_signal.device

        # Sample random t ∼ U(Tmin, Tmax)
        t = torch.randint(
            low=self.t_min, high=self.t_max + 1, size=(B,), device=device
        )

        # Compute new length = length - t (only where t < length / 2)
        new_lengths = torch.where(
            condition=t < (processed_signal_length / 2), 
            input=processed_signal_length - t,
            other=processed_signal_length
        )

        indicies = torch.arange(T, device=device)[None, None, :]
        lengths = new_lengths[:, None, None]

        # Apply TrimTail with masked_fill
        processed_signal = processed_signal.masked_fill_(indicies >= lengths, 0.0)

        return processed_signal, new_lengths


# Custom hyperbolic interpolation
def hyperbolic_interp(w0, w1, num_steps, steepness=5.0, flip=False):
    t = np.linspace(0, 1, num_steps)
    curve = 1 / (1 + steepness * (1 - t)) if flip else 1 / (1 + steepness * t)
    curve = (curve - curve[-1]) / (curve[0] - curve[-1])  # normalize to [0, 1]
    return w1 + (w0 - w1) * curve

class AWP:
    """ 
    Align with purpose implementation 
    Source: https://arxiv.org/pdf/2307.01715v3
    """
    def __init__(self,
                num_samples: int,
                loss_weights: float,
                loss_weight_steps: list,
                lambda_bias: float,
                blank_id: int,
                add_positional_bias=None, # depriacted
            ):
        """
        Args:
            num_samples: number of samples in the batch (N)
            loss_weights: float, array of loss weights 
            loss_weight_steps: list, steps on which weights from loss_weights ar applied
            lambda_bias: float, bias for the loss function
            blank_id: int, index of blank token
        """
        super().__init__()
        self.num_samples = num_samples
        self.lambda_bias = lambda_bias
        self._loss_fn = torch.nn.MultiMarginLoss(
            margin=self.lambda_bias, reduction='mean', 
        )
        self.weight = 0.0
        self.loss_weight_position = 0
        self.blank_id = blank_id
        
        self.loss_weight_steps = []
        self.loss_weights = []

        assert len(loss_weights) == len(loss_weight_steps), "Mismatch in weights/steps"

        for w_spec, s_spec in zip(loss_weights, loss_weight_steps):
            if isinstance(w_spec, Iterable) or isinstance(s_spec, Iterable):
                # interp_type, w0, w1 = w_spec
                # s0, s1 = s_spec
                # if s1 <= s0:
                #     raise ValueError(f"Invalid step range: {s_spec}")
                # steps = list(range(s0, s1 + 1))
                # if interp_type == 'lininterp':
                #     weights = np.linspace(w0, w1, len(steps)).tolist()
                # elif interp_type == 'loginterp':
                #     weights = np.logspace(np.log10(w0), np.log10(w1), len(steps)).tolist()
                # elif interp_type == 'sine':
                #     t = np.linspace(0, np.pi, len(steps))
                #     curve = 0.5 * (1 + np.cos(t))  # symmetric decay from 1 to 0
                #     weights = (w1 + (w0 - w1) * curve).tolist()
                # else:
                #     raise ValueError(f"Unknown interp type: {interp_type}")
                interp_type, w0, w1 = w_spec[:3]
                s0, s1 = s_spec
                if s1 <= s0:
                    raise ValueError(f"Invalid step range: {s_spec}")
                steps = list(range(s0, s1 + 1))

                if interp_type == 'lininterp':
                    weights = np.linspace(w0, w1, len(steps)).tolist()
                elif interp_type == 'loginterp':
                    weights = np.logspace(np.log10(w0), np.log10(w1), len(steps)).tolist()
                elif interp_type == 'sine':
                    t = np.linspace(0, np.pi, len(steps))
                    curve = 0.5 * (1 + np.cos(t))  # symmetric decay from 1 to 0
                    weights = (w1 + (w0 - w1) * curve).tolist()
                elif interp_type == 'hyperbolic':
                    weights = hyperbolic_interp(w0, w1, len(steps), w_spec[3], flip=False)
                elif interp_type == 'hyperbolic_flip':
                    weights = hyperbolic_interp(w0, w1, len(steps), w_spec[3], flip=True)
                else:
                    raise ValueError(f"Unknown interp type: {interp_type}")
                
                assert len(steps) == len(weights)
                self.loss_weight_steps.extend(steps)
                self.loss_weights.extend(weights)
            else:
                self.loss_weight_steps.append(s_spec)
                self.loss_weights.append(float(w_spec))

        # Just to be safe
        zipped = sorted(zip(self.loss_weight_steps, self.loss_weights), key=lambda x: x[0])
        self.loss_weight_steps, self.loss_weights = map(list, zip(*zipped))

    
    def update_weight(self, global_step):
        """
        Update the weight of the awp loss according to schedule
        """
        while self.loss_weight_position < len(self.loss_weight_steps) \
              and global_step >= self.loss_weight_steps[self.loss_weight_position]:
            self.weight = self.loss_weights[self.loss_weight_position]
            # logging.info(f"AWP loss weight updated to {self.weight} at step {global_step}")
            self.loss_weight_position += 1


    def sample_alignments(self, probs):
        """
        probs: tensor of shape (B, T, D)
        alignments: output of size (B, T, N)
        """
        B, T, D = probs.shape

        # Sample indices along the D dimension
        probs_reshaped = probs.reshape(B * T, D)
        alignments = torch.multinomial(
            probs_reshaped, num_samples=self.num_samples, replacement=True,
        )
        return alignments.reshape(B, T, self.num_samples)


    def alignment_logprobability(self, log_probs, alignments, legnth_mask):
        """
        log_probs: tensor of shape (B, T, D)
        alignments: tensor of shape (B, T, N)
        legnth_mask: tensor of shape (B, T, 1)
        output: shape (B, N)
        """
        alignment_loglikes = torch.gather(log_probs, dim=2, index=alignments)
        masked_loglikes = alignment_loglikes * legnth_mask.float()
        # out: shape (B, T, N)
        return masked_loglikes.sum(dim=1)


    @staticmethod
    def get_length_mask(input_legnths, max_length):
        """
        input_lengths: tensor of shape (B)
        output of size (B, T, 1)
        """
        indices = torch.arange(max_length, device=input_legnths.device)
        return indices[None, :, None] < input_legnths[:, None, None]


    def shift_left(self, alignments, input_lengths):
        """
        alignments: tensor of shape (B, T, N)
        input_lengths: tensor of shape (B)
        output of size (B, T, N) - improved alignments
        """
        _, T, _ = alignments.shape
        # -> [B, T, 1]
        mask = self.get_length_mask(input_lengths - 1, T)

        # -> [B, T, N]
        alignments = torch.roll(alignments, shifts=-1, dims=1)
        alignments = alignments * mask + self.blank_id * (~mask)
        return alignments


    def awp_low_latency(self, alignments, input_lengths):
        """
        alignments: tensor of shape (B, T, N); N - num sampled alignments
        input_lengths: tensor of shape (B)
        output: tensor of shape (B, T, N) - improved alignments
        """

        B, T, N = alignments.shape
        device = alignments.device
        
        # -> [T]
        indices = torch.arange(T, device=device)
        
        # -> [B, T, 1]
        length_mask = indices[None, :, None] < input_lengths[:, None, None]
        
        # -> [B, T-1, N]; where the current alignment token is the same as the previous one
        repeats_mask = alignments[:, 1:, :] == alignments[:, :-1, :]
        
        # -> [B, T-1, N]; match both masks and not blank
        candidates = repeats_mask & length_mask[:, 1:, :] & (alignments[:, 1:, :] != self.blank_id)
        
        # -> [B, T-1, N]; punish choices with small positions
        # if self.add_positional_bias:
        #     bias = 1 / torch.linspace(1, 1e-3, T-1, device=device)
        #     candidates = (candidates * bias[None, :, None]) / bias.sum()

        # -> [B, T, N]; pos 0 will be sampled if all else are 0
        candidates = torch.concat(
            [torch.full((B, 1, N), 1e-12, device=device), candidates.float()], dim=1
        )

        # -> [B, N, T]
        candidates = candidates.transpose(1, 2)
        
        # -> [B * N, T]
        candidates = candidates.reshape(B * N, T)
        
        # -> [B * N, 1]
        selected_positions = torch.multinomial(candidates, num_samples=1)
        
        # -> [B, N]
        selected_positions = selected_positions.reshape(B, N)
        
        # -> [B, N];  Don't move if nothing matched.
        candidate_positions = torch.where(selected_positions == 0, input_lengths[:, None], selected_positions)

        # -> [B, T, N]
        shifted_alignments = self.shift_left(alignments, input_lengths)
        mask = indices[None, :, None] < candidate_positions[:, None, :]
        alignments = alignments * mask + shifted_alignments * (~mask)
        return alignments


    def shift_alignments(self, alignments: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """
        Efficiently shifts alignment values in time based on per-batch and per-alignment shift amounts.
        Inserts blanks at the front for positive shifts and at the end for negative shifts.

        Args:
            alignments (torch.Tensor): [B, T, N] tensor
            shifts (torch.Tensor): [B, N] tensor with int values
            blank_id (int): ID representing a blank token

        Returns:
            torch.Tensor: Shifted alignments of shape [B, T, N]
        """
        B, T, N = alignments.shape
        device = alignments.device

        time = torch.arange(T, device=device).view(1, T, 1).expand(B, T, N)
        shifts_expanded = shifts.view(B, 1, N).expand(B, T, N)

        # Shifted indices
        shifted_time = (time - shifts_expanded) % T
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, N)
        feat_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, T, N)

        result = alignments[batch_idx, shifted_time, feat_idx]

        # Mask for padding
        result.masked_fill_((shifts_expanded > 0) & (time < shifts_expanded), self.blank_id)
        result.masked_fill_((shifts_expanded < 0) & (time >= T + shifts_expanded), self.blank_id)
        # result = alignments
        return result


    def align_to_timings(self, alignments, input_lengths, start, end):
        """
        alignments: tensor of shape (B, T, N); N - num sampled alignments
        input_lengths: tensor of shape (B)
        output: tensor of shape (B, T, N) - improved alignments
        start: tensor of shape (B)
        end: tensor of shape (B)
        """

        B, T, N = alignments.shape
        device = alignments.device

        # mask of non-blank tokens: shape [B, T]
        non_blank_mask = (alignments != self.blank_id)

        # -> [T]
        time = torch.arange(T, device=device).view(1, T, 1).expand(B, T, N)

        # -> [B, N]
        first_non_blank = torch.where(
            non_blank_mask, time, torch.full_like(alignments, T)
        ).min(dim=1).values

        # -> [B, N]
        shifts = torch.where(first_non_blank < start.view(B, 1).expand(B, N), 1, 0)
        
        # -> [B, N]
        last_non_blank = torch.where(
            non_blank_mask, time, torch.full_like(alignments, -1)
        ).max(dim=1).values

        # -> [B, N]
        shifts = torch.where(last_non_blank > end.view(B, 1).expand(B, N), -1, shifts)

        return self.shift_alignments(alignments, shifts)
        
        

    def get_loss(self, log_probs, alignments, input_lengths, starts=None, ends=None):
        """
        log_probs: tensor of shape (B, T, D)
        alignments: tensor of shape (B, T, N)
        input_lengths: tensor of shape (B)
        output of size ()
        """

        if starts is not None:
            # -> [B, T, N]
            improved_alignments = self.align_to_timings(alignments, input_lengths, starts, ends)
        else:
            # -> [B, T, N]
            improved_alignments = self.awp_low_latency(alignments, input_lengths)

        # -> [B, T, 1]
        mask = self.get_length_mask(input_lengths, log_probs.shape[1])

        # -> [B, N]
        alignment_proba = self.alignment_logprobability(log_probs, alignments, mask)

        # -> [B, N]
        improved_alignment_proba = self.alignment_logprobability(log_probs, improved_alignments, mask)

        # -> [B * N, 2]
        x = torch.stack([alignment_proba.reshape(-1), improved_alignment_proba.reshape(-1)], dim=1)
        return self.weight * self._loss_fn(x, torch.ones(x.shape[0], device=x.device, dtype=torch.int64))



def ctc_ce_split(log_probs, starts, ends, lens, blank_id):
    """
    Split log_probs into CTC and CE parts.
    CTC: between `starts` and `ends`
    CE: all other valid time steps (before `lens`) excluding the CTC region.

    Args:
        log_probs: Tensor of shape [B, T, D]
        starts: Tensor of shape [B], start indices (inclusive)
        ends: Tensor of shape [B], end indices (exclusive)
        lens: Tensor of shape [B], actual lengths of each sequence

    Returns:
        ctc_log_probs: [B, T_ctc, D]
        ctc_lens: [B]
        ce_log_probs: [N_ce, D] (flattened valid CE entries only)
        ce_lens: [B]
    """
    B, T, D = log_probs.shape
    device = log_probs.device

    # [1, T]
    positions = torch.arange(T, device=device).unsqueeze(0)

    # [B, T] - boolean masks
    ctc_mask = (positions >= starts[:, None]) & (positions < ends[:, None])
    ce_mask = (~ctc_mask) & (positions < lens[:, None])

    # [B] - counts of valid positions
    ctc_lens = ctc_mask.sum(dim=1)
    ce_lens = ce_mask.sum(dim=1)

    # ---- CTC ----
    ctc_pos = torch.cumsum(ctc_mask, dim=1) - 1
    non_ctc_pos = torch.cumsum(~ctc_mask, dim=1) - 1
    ctc_index = torch.where(ctc_mask, ctc_pos, non_ctc_pos + ctc_lens[:, None])
    
    # Expand into 3 dimesions to avoid a bug in torch.scatter
    ctc_index = ctc_index.unsqueeze(-1).expand(-1, -1, D)
    ctc_logprobs = torch.zeros_like(log_probs).scatter(1, ctc_index, log_probs)
    ctc_logprobs = ctc_logprobs[:, :ctc_lens.max()]

    # ---- CE ----
    # [N_ce, D] — gather only valid CE positions
    ce_logprobs = log_probs[ce_mask]
    
    # Use blank targets as dummy supervision (CE as auxiliary regularizer)
    # [N_ce]
    ce_targets = torch.full(
        (ce_logprobs.shape[0],),
        fill_value=blank_id,
        dtype=lens.dtype,
        device=device,
    )
    return ctc_logprobs, ctc_lens, ce_logprobs, ce_lens, ce_targets



class TimingAugmentor:
    def __init__(self,
                 offset: float,
                 min_left_margin_sec: float,
                 max_left_margin_sec: float,
                 min_right_margin_sec: float,
                 max_right_margin_sec: float,
                 frame_size: float
                 ):
        self.offset = offset
        self.min_left_margin_sec = min_left_margin_sec
        self.max_left_margin_sec = max_left_margin_sec
        self.min_right_margin_sec = min_right_margin_sec
        self.max_right_margin_sec = max_right_margin_sec
        self.frame_size = frame_size

    def _sample_time(self, t_min, t_max, size, device):
        if abs(t_min - t_max) < 1e-12:
            return torch.full(size, round(t_min / self.frame_size), device=device)
        else:
            return torch.randint(
                low=floor(t_min / self.frame_size), high=ceil(t_max / self.frame_size) + 1, size=size, device=device
            )

    def __call__(self, starts, ends, lens):
        left_margin_sec = self._sample_time(
            self.min_left_margin_sec, self.max_left_margin_sec, size=starts.shape, device=starts.device
        )
        right_margin_sec = self._sample_time(
            self.min_right_margin_sec, self.max_right_margin_sec, size=ends.shape, device=ends.device
        )
        zeros = 0 * lens
        starts = torch.clip(
            ((self.offset + starts) / self.frame_size).floor() - left_margin_sec, zeros, lens
        )
        ends = torch.clip(
            ((self.offset + ends) / self.frame_size).ceil() + right_margin_sec, zeros, lens
        )
        return starts, ends


class EncDecCTCModel(ASRModel, ExportableEncDecModel, ASRModuleMixin, InterCTCMixin, ASRTranscriptionMixin):
    """Base class for encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = EncDecCTCModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = EncDecCTCModel.from_config_dict(self._cfg.encoder)
    
        # Mel spectrogram -> 10ms / frame
        # Subsampling -> 40ms / frame

        # Frame size of the output ~ 40ms; 
        # Frame size of the input (for processed signal) ~ 10ms

        self.out_frame_size_in_sec = (
            self._cfg.preprocessor.window_stride * self._cfg.encoder.subsampling_factor
        )
        
        logging.info(f"Model frame size in seconds is {self.out_frame_size_in_sec=}")

        with open_dict(self._cfg):
            if "feat_in" not in self._cfg.decoder or (
                not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self._cfg.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.decoder.num_classes < 1 and self.cfg.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.decoder.num_classes, len(self.cfg.decoder.vocabulary)
                    )
                )
                cfg.decoder["num_classes"] = len(self.cfg.decoder.vocabulary)

        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)

        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )
        
        self.ce_weight = None
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.blank_id = self.decoder.num_classes_with_blank - 1

        self.train_with_timings = None
        self.timing_augmentor = None

        if "train_with_timings" in self._cfg:
            self.train_with_timings = self._cfg.train_with_timings
            with open_dict(self._cfg):
                if "method" not in self.train_with_timings:
                    self.train_with_timings["method"] = None

            timing_augmentor_cfg = {}

            if "timing_augmentor" in self.train_with_timings:
                timing_augmentor_cfg = self.train_with_timings.timing_augmentor

            self.timing_augmentor = TimingAugmentor(
                offset=self._cfg.train_ds.left_pad_signal_ms * 1e-3,
                min_left_margin_sec=timing_augmentor_cfg.get("min_left_margin_sec", 0),
                max_left_margin_sec=timing_augmentor_cfg.get("max_left_margin_sec", 0),
                min_right_margin_sec=timing_augmentor_cfg.get("min_right_margin_sec", 0),
                max_right_margin_sec=timing_augmentor_cfg.get("max_right_margin_sec", 0),
                frame_size=self.out_frame_size_in_sec,
            )

        # Latency reduction methods:
        if 'awp' in self._cfg:
            kwargs = OmegaConf.to_container(self._cfg.awp)
            assert 'blank_id' not in kwargs, "blank_id is set automatically"
            logging.info(f"Using AWP with kwargs: {kwargs}")
            self.awp = AWP(blank_id=self.blank_id, **kwargs)
        else:
            self.awp = None

        if 'trimtail' in self._cfg:
            tcfg = self._cfg.trimtail
            logging.info(f"Using TrimTail with Tmin={tcfg.tmin_sec}, Tmax={tcfg.tmax_sec}")
            self.trimtail = TrimTail(
                t_min_sec=tcfg.tmin_sec, t_max_sec=tcfg.tmax_sec, frame_size=self._cfg.preprocessor.window_stride,
            )
        else:
            self.trimtail = None

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecCTCModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = CTCDecoding(self.cfg.decoding, vocabulary=OmegaConf.to_container(self.decoder.vocabulary))

        # Setup metric with decoding strategy
        self.wer = WER(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        # setting up interCTC loss (from InterCTCMixin)
        self.setup_interctc(decoder_name='decoder', loss_name='loss', wer_name='wer')

        # Adapter modules setup (from ASRAdapterModelMixin)
        self.setup_adapters()

    def transcribe(
        self,
        audio: Union[str, List[str], torch.Tensor, np.ndarray, DataLoader],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        timestamps: Optional[bool] = None,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray/tensor audio array or 
                path to a manifest file.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels 
                from multi-channel audio. If set to `'average'`, it performs averaging across channels. 
                Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            timestamps: Optional(Bool): timestamps will be returned if set to True as part of hypothesis 
                object (output.timestep['segment']/output.timestep['word']). Refer to `Hypothesis` class 
                for more details. Default is None and would retain the previous state set by 
                using self.change_decoding_strategy().
            verbose: (bool) whether to display tqdm progress bar
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.

        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as 
            paths2audio_files
        """
        timestamps = timestamps or (override_config.timestamps if override_config is not None else None)
        if timestamps is not None:
            # else retain the decoder state (users can set it using change_decoding_strategy)
            if timestamps or (override_config is not None and override_config.timestamps):
                logging.info(
                    "Timestamps requested, setting decoding timestamps to True. Capture them in Hypothesis object, \
                        with output[idx].timestep['word'/'segment'/'char']"
                )
                return_hypotheses = True
                with open_dict(self.cfg.decoding):
                    self.cfg.decoding.compute_timestamps = True
                    self.cfg.decoding.preserve_alignments = True
                self.change_decoding_strategy(self.cfg.decoding, verbose=False)
            else:  # This is done to ensure the state is preserved when decoding_strategy is set outside
                with open_dict(self.cfg.decoding):
                    self.cfg.decoding.compute_timestamps = self.cfg.decoding.get('compute_timestamps', False)
                    self.cfg.decoding.preserve_alignments = self.cfg.decoding.get('preserve_alignments', False)
                self.change_decoding_strategy(self.cfg.decoding, verbose=False)

        return super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            timestamps=timestamps,
            override_config=override_config,
        )

    def change_vocabulary(self, new_vocabulary: List[str], decoding_cfg: Optional[DictConfig] = None):
        """
        Changes vocabulary used during CTC decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        If new_vocabulary == self.decoder.vocabulary then nothing will be changed.

        Args:

            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
            this is target alphabet.

        Returns: None

        """
        if self.decoder.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.decoder.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config['vocabulary'] = new_vocabulary
            new_decoder_config['num_classes'] = len(new_vocabulary)

            del self.decoder
            self.decoder = EncDecCTCModel.from_config_dict(new_decoder_config)
            del self.loss
            self.loss = CTCLoss(
                num_classes=self.decoder.num_classes_with_blank - 1,
                zero_infinity=True,
                reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            )

            if decoding_cfg is None:
                # Assume same decoding config as before
                decoding_cfg = self.cfg.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(CTCDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

            self.decoding = CTCDecoding(
                decoding_cfg=decoding_cfg, vocabulary=OmegaConf.to_container(self.decoder.vocabulary)
            )

            self.wer = WER(
                decoding=self.decoding,
                use_cer=self._cfg.get('use_cer', False),
                dist_sync_on_step=True,
                log_prediction=self._cfg.get("log_prediction", False),
            )

            # Update config
            with open_dict(self.cfg.decoder):
                self._cfg.decoder = new_decoder_config

            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            ds_keys = ['train_ds', 'validation_ds', 'test_ds']
            for key in ds_keys:
                if key in self.cfg:
                    with open_dict(self.cfg[key]):
                        self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

            logging.info(f"Changed decoder to output to {self.decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig, verbose: bool = True):
        """
        Changes decoding strategy used during CTC decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            verbose: (bool) whether to display logging information
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = CTCDecoding(
            decoding_cfg=decoding_cfg, vocabulary=OmegaConf.to_container(self.decoder.vocabulary)
        )

        self.wer = WER(
            decoding=self.decoding,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        self.decoder.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        if verbose:
            logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')

        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                # During transcription, the model is initially loaded on the CPU.
                # To ensure the correct global_rank and world_size are set,
                # these values must be passed from the configuration.
                global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
                world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
                dataset=LhotseSpeechToTextBpeDataset(
                    tokenizer=make_parser(
                        labels=config.get('labels', None),
                        name=config.get('parser', 'en'),
                        unk_id=config.get('unk_index', -1),
                        blank_id=config.get('blank_index', -1),
                        do_normalize=config.get('normalize_transcripts', False),
                    ),
                    return_cuts=config.get("do_transcribe", False),
                ),
            )

        dataset = audio_to_text_dataset.get_audio_to_text_char_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            preprocessor_cfg=self._cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToCharDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        batch_sampler = None
        if config.get('use_semi_sorted_batching', False):
            if not isinstance(dataset, _AudioTextDataset):
                raise RuntimeError(
                    "Semi Sorted Batch sampler can be used with AudioToCharDataset or AudioToBPEDataset "
                    f"but found dataset of type {type(dataset)}"
                )
            # set batch_size and batch_sampler to None to disable automatic batching
            batch_sampler = get_semi_sorted_batch_sampler(self, dataset, config)
            config['batch_size'] = None
            config['drop_last'] = False
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            sampler=batch_sampler,
            batch_sampler=None,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
            
        if self.trimtail is not None and self.training:
            processed_signal, processed_signal_length = self.trimtail(processed_signal=processed_signal, processed_signal_length=processed_signal_length)

        encoder_output = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoder_output[0]
        encoded_len = encoder_output[1]
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return (
            log_probs,
            encoded_len,
            greedy_predictions,
        )

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid) 
        
        starts, ends = None, None 
        
        if len(batch) == 4:
            signal, signal_len, transcript, transcript_len = batch
        elif len(batch) == 6:
            signal, signal_len, transcript, transcript_len, starts, ends = batch
        
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        
        
        if self.train_with_timings is not None:

            starts, ends = self.timing_augmentor(starts, ends, encoded_len)
            
            # if random.randint(0, 100) == 0:
            #     logging.info(
            #         f"starts: {[s * self.out_frame_size_in_sec for s in starts.cpu().numpy().tolist()]}\n" + f"ends: {[s * self.out_frame_size_in_sec for s in (encoded_len - ends).cpu().numpy().tolist()]}"
            #     )
            
            if self.train_with_timings.method == "cross_entropy":
                
                # Perform the logprob split
                ctc_logprobs, ctc_lens, ce_logprobs, ce_lens, ce_targets = ctc_ce_split(log_probs, starts, ends, encoded_len, self.blank_id)

                # Compute CTC loss on ctc_logprobs
                loss_value = self.loss(
                    log_probs=ctc_logprobs,
                    targets=transcript,
                    input_lengths=ctc_lens,
                    target_lengths=transcript_len,
                )

                # Compute CE loss if there are valid CE positions
                if (ce_logprobs.shape[0] > 0):
                    
                    # Repeat each value `a[i]` times
                    # Example: [1, 0, 3, 2] -> [1, 3, 3, 3, 2, 2]
                    # [N_ce]
                    ce_norm = ce_lens.repeat_interleave(ce_lens)
                    
                    # [N_ce, D]
                    ce_loss = torch.nn.functional.cross_entropy(
                        ce_logprobs, ce_targets, reduction="none"
                    )

                    # []
                    ce_loss = (ce_loss / ce_norm).sum() / log_probs.shape[0]

                    loss_value += ce_loss * self.ce_weight

            # elif self.train_with_timings.method == "early_late_pentalties":
                
            #     T = log_probs.shape[1]
                
            #     alpha = 1e-3
            #     t_buffer = 0.040
                
            #     eos_id = self.eos_id
                
            #     pos = torch.arange(0, T, device=encoded_len.device)
                
            #     early_penalty = torch.maximum(0, (ends / self.out_frame_size_in_sec).ceil() - pos)
            #     late_penalty = torch.maximum(0, pos - ((ends + t_buffer) / self.out_frame_size_in_sec).ceil())
                
            #     # [B, T, D]
            #     log_probs[:, :, eos_id] -= early_penalty * alpha
            #     log_probs[:, :, eos_id] -= late_penalty * alpha
                
            #     # Compute CTC loss on ctc_logprobs
            #     loss_value = self.loss(
            #         log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            #     )

            # else:
            #     raise NotImplementedError(f"Unknown train_with_timings.method: {self.train_with_timings.method}")
            else:
                loss_value = self.loss(
                    log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
                )
        else:
            loss_value = self.loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )

        if self.awp is not None:
            probs = log_probs.exp()
            self.awp.update_weight(global_step=self.trainer.global_step)
            alignments = self.awp.sample_alignments(probs)
            loss_value += self.awp.get_loss(
                log_probs=log_probs, alignments=alignments, input_lengths=encoded_len, starts=starts, ends=ends
            )

        # Add auxiliary losses, if registered
        loss_value = self.add_auxiliary_losses(loss_value)
        # only computing WER when requested in the logs (same as done for final-layer WER below)
        loss_value, tensorboard_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=((batch_nb + 1) % log_every_n_steps == 0)
        )

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs.update(
            {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )

        if (batch_nb + 1) % log_every_n_steps == 0:
            self.wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self.wer.compute()
            self.wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, sample_id = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        transcribed_texts = self.wer.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs,
            decoder_lengths=encoded_len,
            return_hypotheses=False,
        )

        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, transcribed_texts))

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        # signal, signal_len, transcript, transcript_len = batch
        starts, ends = None, None
        
        if len(batch) == 4:
            signal, signal_len, transcript, transcript_len = batch

        elif len(batch) == 6:
            signal, signal_len, transcript, transcript_len, starts, ends = batch


        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        loss_value, metrics = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            log_wer_num_denom=True,
            log_prefix="val_",
        )

        self.wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        self.wer.reset()
        metrics.update({'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom, 'val_wer': wer})

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        head_delay, tail_delay = numba_calculate_expected_delays(
            log_probs.detach().cpu().numpy(), input_lengths=encoded_len.cpu().numpy(), blank_id=self.blank_id
        )

        metrics.update({
            "head_nonblank_delay_sec": torch.tensor(self.out_frame_size_in_sec * head_delay, dtype=torch.float32, device=loss_value.device),
            "tail_nonblank_delay_sec": torch.tensor(self.out_frame_size_in_sec * tail_delay, dtype=torch.float32, device=loss_value.device),
        })
        
        if starts is not None:
            
            # # log starts, ends and head_delay and tail_delay
            
            # # 1. log starts
            # logging.info(
            #     f"starts: {[s for s in starts.tolist()]}\n"
            # )
            # logging.info(
            #     f"ends: {[e for e in ends.tolist()]}\n"
            # )
            # logging.info(
            #     f"head_delay: {[self.out_frame_size_in_sec * d for d in head_delay.tolist()]}\n"
            # )
            # logging.info(
            #     f"tail_delay: {[self.out_frame_size_in_sec * d for d in tail_delay.tolist()]}\n"
            # )
            
            offset = self._cfg.test_ds.left_pad_signal_ms * 1e-3
            starts = offset + starts.cpu().numpy()
            ends = offset + ends.cpu().numpy()

            metrics.update({
                "start_diff": torch.tensor(self.out_frame_size_in_sec * head_delay - starts, dtype=torch.float32, device=loss_value.device),
                "end_diff": torch.tensor(self.out_frame_size_in_sec * tail_delay - ends, dtype=torch.float32, device=loss_value.device),
            })

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        metrics = super().multi_validation_epoch_end(outputs, dataloader_idx)
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")
        return metrics

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        metrics = super().multi_test_epoch_end(outputs, dataloader_idx)
        self.finalize_interctc_metrics(metrics, outputs, prefix="test_")
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    """ Transcription related methods """

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        logits, logits_len, greedy_predictions = self.forward(input_signal=batch[0], input_signal_length=batch[1])
        output = dict(logits=logits, logits_len=logits_len)
        del greedy_predictions
        return output

    def _transcribe_output_processing(self, outputs, trcfg: TranscribeConfig) -> GenericTranscriptionType:
        logits = outputs.pop('logits')
        logits_len = outputs.pop('logits_len')

        hypotheses = self.decoding.ctc_decoder_predictions_tensor(
            logits,
            decoder_lengths=logits_len,
            return_hypotheses=trcfg.return_hypotheses,
        )
        if trcfg.return_hypotheses:
            if logits.is_cuda:
                # See comment in
                # ctc_greedy_decoding.py::GreedyCTCInfer::forward() to
                # understand this idiom.
                logits_cpu = torch.empty(logits.shape, dtype=logits.dtype, device=torch.device("cpu"), pin_memory=True)
                logits_cpu.copy_(logits, non_blocking=True)
            else:
                logits_cpu = logits
            logits_len = logits_len.cpu()
            # dump log probs per file
            for idx in range(logits_cpu.shape[0]):
                # We clone because we don't want references to the
                # cudaMallocHost()-allocated tensor to be floating
                # around. Were that to be the case, then the pinned
                # memory cache would always miss.
                hypotheses[idx].y_sequence = logits_cpu[idx, : logits_len[idx]].clone()
                if hypotheses[idx].alignments is None:
                    hypotheses[idx].alignments = hypotheses[idx].y_sequence
            del logits_cpu

        # cleanup memory
        del logits, logits_len

        if trcfg.timestamps:
            hypotheses = process_timestamp_outputs(
                hypotheses, self.encoder.subsampling_factor, self.cfg['preprocessor']['window_stride']
            )

        return hypotheses

    def get_best_hyptheses(self, all_hypothesis: list[list[Hypothesis]]):
        return [hyp[0] for hyp in all_hypothesis]

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': OmegaConf.to_container(self.decoder.vocabulary),
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'channel_selector': config.get('channel_selector', None),
            'left_pad_signal_ms': config.get('left_pad_signal_ms', None),
            'right_pad_signal_ms': config.get('right_pad_signal_ms', None),
        }
        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5Base-En",
            description="QuartzNet15x5 model trained on six datasets: LibriSpeech, Mozilla Common Voice \
                (validated clips from en_1488h_2019-12-10), WSJ, Fisher, Switchboard, and NSC Singapore English. \
                    It was trained with Apex/Amp optimization level O1 for 600 epochs. The model achieves a WER of \
                    3.79% on LibriSpeech dev-clean, and a WER of 10.05% on dev-other. Please visit \
                        https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels for further details.",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_quartznet15x5/versions/1.0.0rc1/files/stt_en_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_jasper10x5dr",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_jasper10x5dr",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_jasper10x5dr/versions/1.0.0rc1/files/stt_en_jasper10x5dr.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ca_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ca_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ca_quartznet15x5/versions/1.0.0rc1/files/stt_ca_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_it_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_it_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_quartznet15x5/versions/1.0.0rc1/files/stt_it_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_fr_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_quartznet15x5/versions/1.0.0rc1/files/stt_fr_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_quartznet15x5/versions/1.0.0rc1/files/stt_es_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_quartznet15x5/versions/1.0.0rc1/files/stt_de_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_pl_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_pl_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_pl_quartznet15x5/versions/1.0.0rc1/files/stt_pl_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ru_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ru_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_quartznet15x5/versions/1.0.0rc1/files/stt_ru_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_citrinet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_zh_citrinet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_citrinet_512/versions/1.0.0rc1/files/stt_zh_citrinet_512.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_zh_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_citrinet_1024_gamma_0_25/versions/1.0.0/files/stt_zh_citrinet_1024_gamma_0_25.nemo",
        )

        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_zh_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_citrinet_1024_gamma_0_25/versions/1.0.0/files/stt_zh_citrinet_1024_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="asr_talknet_aligner",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:asr_talknet_aligner",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/asr_talknet_aligner/versions/1.0.0rc1/files/qn5x5_libri_tts_phonemes.nemo",
        )
        results.append(model)

        return results

    @property
    def adapter_module_names(self) -> List[str]:
        return ['', 'encoder', 'decoder']

    @property
    def wer(self):
        return self._wer

    @wer.setter
    def wer(self, wer):
        self._wer = wer
