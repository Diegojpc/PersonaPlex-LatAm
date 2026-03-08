# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download

# Local imports
from moshi.models import loaders
from moshi.dataset import SpanishMoshiDataset, collate_fn

# PEFT for LoRA
try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    print("Please install peft: pip install peft")
    raise

logger = logging.getLogger(__name__)

def setup_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def train(args):
    setup_logger()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Mimi (for on-the-fly tokenization)
    logger.info("Loading Mimi...")
    mimi_weight = args.mimi_weight
    if mimi_weight is None:
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.eval()
    mimi.requires_grad_(False) # Freeze Mimi
    
    # 2. Load Moshi LM
    logger.info("Loading Moshi LM...")
    moshi_weight = args.moshi_weight
    if moshi_weight is None:
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    
    # We load onto CPU first to apply LoRA, then move to GPU to save memory if possible
    lm_model = loaders.get_moshi_lm(moshi_weight, device="cpu", dtype=torch.float32) 
    
    # 3. Apply LoRA
    logger.info("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=None, # Custom model
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["linear", "text_linear", "depformer_in", "linears"] # Target linear layers in Transformer
        # Note: 'linear' might match too broadly or not enough, ideally target 'q_proj', 'v_proj' etc if standard transformer.
        # Moshi uses 'text_linear' and 'linears' (list). 
        # Detailed targeting might be safer:
        # target_modules = ["out_proj", "text_linear"] + [f"linears.{i}" for i in range(8)]
    )
    
    # Enable gradients for LoRA only
    lm_model = get_peft_model(lm_model, peft_config)
    lm_model.print_trainable_parameters()
    lm_model.to(device)
    
    # 4. Dataset & Dataloader
    logger.info("Loading Dataset...")
    dataset = SpanishMoshiDataset(
        manifest_path=args.manifest_path,
        tokenizer_path=args.tokenizer_path if args.tokenizer_path else hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME),
        mimi_model=mimi,
        sample_rate=mimi.sample_rate,
        max_duration=args.max_duration,
        device=device
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 5. Optimizer
    optimizer = torch.optim.AdamW(lm_model.parameters(), lr=args.lr)
    
    # 6. Training Loop
    lm_model.train()
    step = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Starting Epoch {epoch+1}/{args.epochs}")
        pbar = tqdm(dataloader)
        
        for batch in pbar:
            codes = batch['codes'].to(device) # [B, 9, T]
            # mask = batch['mask'].to(device) # Not always used in forward_train but good to have
            
            # Forward Pass
            # forward_train expects [B, K, T] where K=9 (1 text + 8 audio)
            # It internally handles shifting for next-token prediction
            output = lm_model.forward_train(codes)
            
            # Loss Calculation
            # output has:
            # - logits: [B, 8, T, audio_card] (Audio prediction)
            # - text_logits: [B, 1, T, text_card] (Text prediction)
            # Targets are the shifted input codes (handled inside or outside? 
            # forward_train documentation says: 
            # "returns LMOutput(logits, logits_mask, text_logits, text_logits_mask)"
            # And it calculates logits for the 'next' token.
            # So targets should be the codes starting from position 1? 
            # Actually forward_train documentation in lm.py says:
            # "map back the logits... to logits on original codes... and provide the corresponding mask over invalid positions"
            # It seems it returns logits ALIGNED with the input codes `codes`.
            # So we supervise `codes` with `logits`.
            
            loss_audio = 0
            for k in range(8):
                # logits: [B, 8, T, card]
                # codes: [B, 9, T]. Audio starts at index 1.
                # So audio channel k is codes[:, k+1]
                
                channel_logits = output.logits[:, k, :, :] # [B, T, card]
                channel_targets = codes[:, k+1, :]     # [B, T]
                channel_mask = output.mask[:, k, :]    # [B, T] valid positions
                
                # Flatten
                flat_logits = channel_logits.reshape(-1, channel_logits.shape[-1])
                flat_targets = channel_targets.reshape(-1)
                flat_mask = channel_mask.reshape(-1)
                
                # Apply mask (or ignore_index=-1)
                # LMModel.zero_token_id is -1.
                
                loss_k = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
                loss_audio += loss_k
            
            loss_audio = loss_audio / 8.0
            
            # Text Loss
            text_logits = output.text_logits[:, 0, :, :] # [B, T, text_card]
            text_targets = codes[:, 0, :]             # [B, T]
            text_mask = output.text_mask[:, 0, :]
            
            flat_text_logits = text_logits.reshape(-1, text_logits.shape[-1])
            flat_text_targets = text_targets.reshape(-1)
            flat_text_mask = text_mask.reshape(-1)
            
            loss_text = F.cross_entropy(flat_text_logits[flat_text_mask], flat_text_targets[flat_text_mask])
            
            # Combined Loss
            total_loss = loss_text + loss_audio
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Loss: {total_loss.item():.4f} (Text: {loss_text.item():.3f}, Audio: {loss_audio.item():.3f})")
            
            step += 1
            
            if step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                lm_model.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")

    # Final Save
    lm_model.save_pretrained(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to JSON dataset manifest")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--mimi_weight", type=str, default=None)
    parser.add_argument("--moshi_weight", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_duration", type=float, default=10.0)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
