import os
import sys
import json
import uuid
import logging
import traceback
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load env vars from .env file
load_dotenv()

def setup_rigorous_logger(name: str) -> logging.Logger:
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
    log_dir.mkdir(exist_ok=True, parents=True)
    fh = logging.FileHandler(log_dir / "personaplex_baseline_eval.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

log = setup_rigorous_logger("BaselineEvaluator")

class PersonaPlexBaselineEval:
    def __init__(self, work_dir: str):
        log.info(f"Initializing PersonaPlex Baseline Evaluator in workspace: {work_dir}")
        self.work_dir = Path(work_dir)
        
        if not self.work_dir.exists():
            log.warning(f"Working directory {self.work_dir} missing. Creating it.")
            self.work_dir.mkdir(parents=True, exist_ok=True)
            
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            log.critical("HF_TOKEN environment variable not set. Halting execution to prevent authentication failure.")
            raise ValueError("Missing HF_TOKEN")
            
        self.env = os.environ.copy()

    def run_inference_turn(self, input_wav: str, prompt_text: str, voice_prompt: str = "NATF2.pt") -> Optional[Dict[str, Any]]:
        run_id = str(uuid.uuid4())[:8]
        output_wav = self.work_dir / f"eval_output_{run_id}.wav"
        output_json = self.work_dir / f"eval_output_{run_id}.json"
        
        log.info(f"[RunID:{run_id}] INITIALIZING INFERENCE TURN")
        log.debug(f"[RunID:{run_id}] Parameters -> input_wav: {input_wav}, "
                  f"prompt_length: {len(prompt_text)} chars, voice_cond: {voice_prompt}")

        if not Path(input_wav).exists():
            log.error(f"[RunID:{run_id}] CRITICAL: Input WAV file not found at path: {input_wav}")
            return None

        # Build execution command
        # Using sys.executable to ensure the same Python environment is used
        cmd = [
            sys.executable, "-m", "moshi.offline",
            "--voice-prompt", voice_prompt,
            "--text-prompt", prompt_text,
            "--input-wav", input_wav,
            "--seed", "42424242",
            "--output-wav", str(output_wav),
            "--output-text", str(output_json),
            "--cpu-offload" 
        ]
        
        log.debug(f"[RunID:{run_id}] Execution Command: {' '.join(cmd)}")
        
        try:
            # We must run this from the project root where `moshi` module is available.
            project_root = str(Path(__file__).parent.parent)
            log.info(f"[RunID:{run_id}] Executing inference from root: {project_root}")
            
            result = subprocess.run(
                cmd, env=self.env, cwd=project_root, capture_output=True, text=True, check=True
            )
            log.info(f"[RunID:{run_id}] SUCCESS: Inference execution completed without OS errors.")
            log.debug(f"[RunID:{run_id}] Engine STDOUT Extract: {result.stdout[-500:]}")
            
            if output_json.exists() and output_wav.exists():
                with open(output_json, 'r') as f:
                    eval_data = json.load(f)
                log.info(f"[RunID:{run_id}] VERIFIED: Output JSON and WAV state persisted accurately.")
                return eval_data
            else:
                log.error(f"[RunID:{run_id}] ARCHITECTURE FAILURE: Subprocess returned 0, but output files {output_json} are missing.")
                return None
                
        except subprocess.CalledProcessError as e:
            log.error(f"[RunID:{run_id}] EXECUTION FAILED: Subprocess returned error code {e.returncode}.")
            log.error(f"[RunID:{run_id}] Engine STDERR Frame: \n{e.stderr[-2000:]}")
            return None
        except Exception as e:
            log.critical(f"[RunID:{run_id}] UNHANDLED SYSTEM EXCEPTION during inference.")
            log.critical(traceback.format_exc())
            return None

if __name__ == "__main__":
    try:
        # Directory where outputs will be dumped
        evaluator = PersonaPlexBaselineEval(work_dir=str(Path(__file__).parent / "baseline_outputs"))
        
        spanish_prompt = (
            "Eres un agente de soporte técnico en Bogotá. Tu nombre es Santiago. "
            "Información: Tienes que ayudar a revisar por qué el internet está tan lento últimamente. "
            "Cobran 80,000 pesos por la revisión presencial."
        )
        
        # NOTE: The user MUST provide this file!
        test_wav_path = str(Path(__file__).parent / "input_colombia.wav")
        
        if not Path(test_wav_path).exists():
           log.error(f"CRITICAL: User has not provided the audio file at {test_wav_path}.")
           log.error("Please place a test audio file mapping to an actual Colombian speaker in this path, then re-run.")
           sys.exit(1)
           
        result = evaluator.run_inference_turn(test_wav_path, spanish_prompt, voice_prompt="NATM1.pt")
        
        if result:
            log.info("Baseline turn finalized successfully. Evaluation outputs generated in baseline_outputs directory.")
            # Optionally print out the transcription tokens that moshi recognized
            if 'text' in result:
                log.info(f"Recognized Text (ASR output from model text tokens): {result['text']}")
        else:
            log.error("Baseline execution collapsed. Review the logs above.")

    except Exception as general_err:
        log.critical("Application-level critical error halted execution initialization.")
        log.critical(traceback.format_exc())
        sys.exit(1)
