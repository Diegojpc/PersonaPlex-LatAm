# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create a JSON manifest for training")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing audio and text files")
    parser.add_argument("--output", type=str, default="train_manifest.json", help="Output JSON path")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    samples = []
    
    # Simple strategy: Look for wav files and matching txt files
    for wav_file in data_dir.rglob("*.wav"):
        txt_file = wav_file.with_suffix(".txt")
        if not txt_file.exists():
             # Try replacing extension strictly
             txt_file = wav_file.parent / (wav_file.stem + ".txt")
        
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                
            samples.append({
                "audio_path": str(wav_file.absolute()),
                "text": text,
                # "codes_path": optional if you pre-compute
            })
    
    print(f"Found {len(samples)} valid samples.")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Manifest saved to {args.output}")

if __name__ == "__main__":
    main()
