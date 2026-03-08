# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import torchaudio
import sentencepiece
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import logging
import sphn

logger = logging.getLogger(__name__)

from .models.compression import MimiModel

class SpanishMoshiDataset(torch.utils.data.Dataset):
    """
    Dataset for fine-tuning Moshi on Spanish audio/text data.
    
    Expects a JSON manifest file or a directory structure with aligned .wav and .txt/.json files.
    The format assumes we have:
    - audio_path: str
    - text: str (transcript)
    - alignment: Optional[List[Dict]] (start, end, word/segment) for precise timing
    """
    def __init__(
        self, 
        manifest_path: str, 
        tokenizer_path: str, 
        mimi_model: Optional[MimiModel] = None,
        sample_rate: int = 24000, 
        frame_rate: int = 25, # Moshi frame rate 12.5 or 25 depending on setup, need to verify
        max_duration: float = 30.0,
        device: str = "cpu"
    ):
        super().__init__()
        self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.max_duration = max_duration
        self.device = device
        self.mimi_model = mimi_model
        
        # Load samples
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
            
    def _load_audio(self, path: str) -> torch.Tensor:
        """Loads and resamples audio to target sample_rate."""
        # Using torchaudio for convenience in training loop, but sphn is also fine
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True) # Mix to mono
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav)
        return wav

    def _get_mimi_codes(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Encodes audio waveform into discrete codes using Mimi.
        Returns tensor of shape [K, T] where K is num_codebooks.
        """
        if self.mimi_model is None:
            raise ValueError("MimiModel must be provided to encode audio on-the-fly.")
            
        with torch.no_grad():
            wav = wav.to(self.device).unsqueeze(0) # Add batch: [1, 1, T]
            codes = self.mimi_model.encode(wav) # [1, K, T_frames]
            return codes.squeeze(0).cpu()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio_path = item['audio_path']
        transcript = item['text']
        
        # 1. Process Text
        # System tag wrapper might be needed depending on behavior
        # transcript = f"<system> {transcript} <system>" 
        text_ids = self.tokenizer.encode(transcript, out_type=int)
        
        # 2. Process Audio
        wav = self._load_audio(audio_path)
        
        # Truncate or Pad
        max_samples = int(self.max_duration * self.sample_rate)
        if wav.shape[1] > max_samples:
            wav = wav[:, :max_samples]
            
        # 3. Get Audio Codes (if Mimi is loaded, otherwise assume pre-computed)
        # For efficiency in training, it's better to pre-compute codes, 
        # but for this implementation we support on-the-fly.
        if self.mimi_model:
            audio_codes = self._get_mimi_codes(wav) # [K, T]
        elif 'codes_path' in item:
            audio_codes = torch.load(item['codes_path'])
        else:
            raise ValueError("No Mimi model provided and no pre-computed codes found.")
            
        # 4. Alignment Logic (Simplified)
        # Moshi expects [Text_Channel, Audio_Channel_1, ..., Audio_Channel_8]
        # Text tokens need to be placed 'near' their corresponding audio time.
        # Without ASR timestamps, we can heuristic: spread text evenly or prepend.
        # For strict training, we pad text with the PAD token (3).
        
        T = audio_codes.shape[-1]
        K = audio_codes.shape[0]
        
        # Create the [9, T] tensor
        # Channel 0: Text
        # Channel 1-8: Audio
        
        # Heuristic alignment: Spaced out if no timestamps
        # If timestamps exist in 'alignment', use them.
        
        text_channel = torch.full((T,), 3, dtype=torch.long) # 3 is assumed PAD/EPAD
        
        # Simple heuristic: Prepend text (User prompt style) or distribute?
        # If it's the assistant SPEAKING, the text should align with audio generation.
        # If we act as if we are generating this speech:
        # We can try to distribute text tokens proportionally.
        
        if len(text_ids) > 0:
            if len(text_ids) > T:
                 text_ids = text_ids[:T] # Truncate text if audio too short
            
            # Linear distribution of text tokens across time T
            indices = np.linspace(0, T-1, len(text_ids), dtype=int)
            for i, txt_id in zip(indices, text_ids):
                text_channel[i] = txt_id

        # Combine
        # audio_codes is [8, T]
        # text_channel is [T] -> [1, T]
        
        combined = torch.cat([text_channel.unsqueeze(0), audio_codes], dim=0) # [9, T]
        
        return {
            "codes": combined,
            "mask": torch.ones(T, dtype=torch.bool) # Valid mask
        }

def collate_fn(batch):
    """ pads batch to max length """
    # Simple padding logic
    max_len = max([b['codes'].shape[1] for b in batch])
    
    batched_codes = []
    masks = []
    
    for b in batch:
        c = b['codes'] # [9, T]
        l = c.shape[1]
        pad_len = max_len - l
        
        if pad_len > 0:
            # Pad value for text is 3, for audio usually -1 (or silence code) if ignored
            # For simplicity let's pad with 0 or last code?
            # Moshi likely uses specific padding. 
            # LMModel.zero_token_id is -1.
            pad_tensor = torch.full((9, pad_len), -1, dtype=c.dtype)
            pad_tensor[0, :] = 3 # Text Pad
            c_padded = torch.cat([c, pad_tensor], dim=1)
            mask_padded = torch.cat([b['mask'], torch.zeros(pad_len, dtype=torch.bool)], dim=0)
        else:
            c_padded = c
            mask_padded = b['mask']
            
        batched_codes.append(c_padded)
        masks.append(mask_padded)
        
    return {
        "codes": torch.stack(batched_codes), # [B, 9, T]
        "mask": torch.stack(masks)         # [B, T]
    }
