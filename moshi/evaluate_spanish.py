# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import soundfile as sf
import argparse
import logging
from huggingface_hub import hf_hub_download
import numpy as np

# Local
from moshi.models import loaders, LMGen
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint directory")
    parser.add_argument("--output_wav", type=str, default="output_spanish.wav", help="Output wav file")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Mimi
    print("Loading Mimi...")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.eval()
    
    # 2. Load Moshi LM (Base)
    print("Loading Moshi Base...")
    moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    lm_model = loaders.get_moshi_lm(moshi_weight, device=device, dtype=torch.float32)
    
    # 3. Load LoRA Adapters
    print(f"Loading LoRA adapters from {args.checkpoint}...")
    lm_model = PeftModel.from_pretrained(lm_model, args.checkpoint)
    lm_model.eval()
    lm_model.to(device)
    
    # 4. Generate
    # We use LMGen wrapper
    lm_gen = LMGen(
        lm_model, 
        device=device,
        sample_rate=mimi.sample_rate,
        frame_rate=mimi.frame_rate
    )
    
    # Text tokenizer for input prompt
    tokenizer_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor(tokenizer_path)
    
    prompt_spanish = "Hola, eres un asistente útil. ¿Cómo estás?"
    text_ids = sp.encode(prompt_spanish, out_type=int)
    
    # Need to verify how LMGen expects prompts. 
    # Usually it's fed into the state.
    # For offline generation, we can use a simpler loop or re-use offline.py logic.
    # Here we do a simple loop manually or use the `lm_gen` streaming state.
    
    print(f"Generating for prompt: '{prompt_spanish}'")
    
    # Prepare tokens
    # Moshi expects a text token at every step? No, it's sparse.
    # But for prompt conditioning, we can perform teacher forcing or just insert them.
    
    # Simplified generation loop using LMGen
    # This is complex because Moshi is streaming.
    # We will simulate a stream of silence for user audio, and provide text prompt.
    
    lm_gen.streaming_forever(batch_size=1)
    
    # Feed prompt
    # In server.py, text prompt is encoded and stored in lm_gen.text_prompt_tokens? No, server handles it.
    # LMGen has `text_prompt_tokens` attribute.
    lm_gen.text_prompt_tokens = text_ids
    
    all_codes = []
    
    # Feed 5 seconds of silence (warmup/input)
    # 12.5 Hz * 5s = ~63 frames
    
    print("Running generation...")
    for i in range(100): # Generate ~8 seconds
        # Fake input audio (silence)
        # Mimi encode silence
        chunk = torch.zeros(1, 1, int(mimi.sample_rate / mimi.frame_rate), device=device)
        codes = mimi.encode(chunk) # [1, 8, 1]
        
        # Step
        tokens = lm_gen.step(codes)
        
        if tokens is not None:
             # tokens: [B, 9] (1 text + 8 audio)
             # Extract audio codes [:, 1:]
             audio_tokens = tokens[:, 1:] 
             all_codes.append(audio_tokens)
             
             # Print text if generated
             txt = tokens[0, 0].item()
             if txt not in (0, 3):
                 print(sp.decode([txt]), end="", flush=True)

    print("\nDecoding audio...")
    if len(all_codes) > 0:
        full_codes = torch.cat(all_codes, dim=0).transpose(0, 1).unsqueeze(0) # [1, 8, T]
        # Decode
        wav = mimi.decode(full_codes)
        wav = wav.squeeze().cpu().detach().numpy()
        sf.write(args.output_wav, wav, mimi.sample_rate)
        print(f"Saved to {args.output_wav}")
    else:
        print("No audio generated.")

if __name__ == "__main__":
    main()
