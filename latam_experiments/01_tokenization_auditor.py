import os
import sys
import logging
from pathlib import Path
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load env vars from .env file
load_dotenv()

def setup_rigorous_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [Line:%(lineno)d] - %(message)s"
    )
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

log = setup_rigorous_logger("TokenizerAuditor", "tokenizer_audit.log")

def main():
    log.info("Starting Tokenization Audit for Moshi/PersonaPlex")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        log.critical("HF_TOKEN environment variable is missing. Authentication will fail.")
        sys.exit(1)
        
    repo_id = "nvidia/personaplex-7b-v1"
    filename = "tokenizer_spm_32k_3.model"
    
    try:
        log.info(f"Retrieving tokenizer {filename} from {repo_id}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
        log.info(f"Tokenizer loaded successfully from {model_path}")
    except Exception as e:
        log.critical("Failed to download or locate the tokenizer model from HuggingFace.")
        log.error(str(e))
        sys.exit(1)
        
    try:
        sp = spm.SentencePieceProcessor(model_file=model_path)
    except Exception as e:
        log.critical("SentencePieceProcessor failed to initialize.")
        log.error(str(e))
        sys.exit(1)
        
    input_file = Path(__file__).parent / "input_colombia.txt"
    if not input_file.exists():
        log.error(f"Test corpus {input_file} not found.")
        sys.exit(1)
        
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    total_words = 0
    total_tokens = 0
    
    log.info(f"Auditing {len(lines)} lines of Colombian Spanish text...")
    
    for i, line in enumerate(lines):
        words = line.split()
        word_count = len(words)
        
        try:
            tokens = sp.encode(line)
            token_count = len(tokens)
            
            total_words += word_count
            total_tokens += token_count
            
            local_fertility = token_count / word_count if word_count > 0 else 0
            if local_fertility > 3.0:
                log.warning(f"HIGH FRAGMENTATION (Fertility: {local_fertility:.2f}) -> '{line}'")
                
        except Exception as e:
            log.error(f"Failed to tokenize line {i}: '{line}'. Error: {e}")
            
    if total_words == 0:
        log.warning("Calculated 0 words. Corpus might be empty.")
        sys.exit(0)
        
    overall_fertility = total_tokens / total_words
    log.info(f"=== TOKENIZATION AUDIT RESULTS ===")
    log.info(f"Total Words: {total_words}")
    log.info(f"Total Tokens: {total_tokens}")
    log.info(f"Overall Fertility Rate (Tokens/Word): {overall_fertility:.3f}")
    
    if overall_fertility > 2.5:
        log.error("FERTILITY FAILED: Rate > 2.5. The tokenizer is highly inefficient for this dialect.")
        log.error("A vocabulary expansion process (CPT) with explicit tokenizer surgery is MANDATORY.")
    elif overall_fertility > 1.8:
        log.warning("FERTILITY SUBOPTIMAL: Rate > 1.8. Inference latency will be degraded.")
    else:
        log.info("FERTILITY ACCEPTABLE: The current English-biased tokenizer can handle this representation reasonably.")

if __name__ == "__main__":
    main()
