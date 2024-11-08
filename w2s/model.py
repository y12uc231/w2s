from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Define model-specific LoRA modules
GPT2_LORA_MODULES = ["c_attn", "c_proj"]
LLAMA_LORA_MODULES = [
    "gate_proj",
    "down_proj",
    "up_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

# Model architecture to LoRA modules mapping
ARCHITECTURE_TO_LORA_MODULES = {
    "gpt2": GPT2_LORA_MODULES,
    "gpt2-large": GPT2_LORA_MODULES,
    "gpt2-xl": GPT2_LORA_MODULES,
    "llama": LLAMA_LORA_MODULES,
    "mistral": LLAMA_LORA_MODULES,
    "qwen": LLAMA_LORA_MODULES,
    "gemma": LLAMA_LORA_MODULES,
}

@dataclass
class PredictorConfig(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        ...

@dataclass
class ModelConfig(PredictorConfig):
    name: str
    enable_lora: bool
    lora_modules: Optional[List[str]] = None
    
    def to_dict(self):
        return vars(self)

class AutoCastingScore(torch.nn.Module):
    def __init__(
        self, score: torch.nn.Linear, output_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(score.weight.to(torch.float32).data)
        self.output_dtype = output_dtype
        
    def forward(self, hiddens):
        return torch.nn.functional.linear(
            hiddens.to(self.weight.dtype), self.weight, None
        ).to(self.output_dtype)

def get_model_architecture(model_name: str) -> str:
    """Determine model architecture from name."""
    model_name = model_name.lower()
    
    if "gpt2-xl" in model_name:
        return "gpt2-xl"
    elif "gpt2-large" in model_name:
        return "gpt2-large"
    elif "gpt2" in model_name:
        return "gpt2"
    elif "llama" in model_name:
        return "llama"
    elif "mistral" in model_name:
        return "mistral"
    elif "qwen" in model_name:
        return "qwen"
    elif "gemma" in model_name:
        return "gemma"
    else:
        return "gpt2"  # default to GPT2 modules if unknown

def init_tokenizer(cfg: ModelConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def init_model(tokenizer, cfg: ModelConfig):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.name,
        torch_dtype="auto",
        device_map={"": "cuda"},
    )
    
    if cfg.lora_modules is None and cfg.enable_lora:
        # Get architecture-specific LoRA modules
        architecture = get_model_architecture(cfg.name)
        cfg.lora_modules = ARCHITECTURE_TO_LORA_MODULES.get(architecture, GPT2_LORA_MODULES)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.score.weight.data *= 0.01
    model.config.problem_type = "single_label_classification"
    
    if cfg.enable_lora:
        # Get architecture-specific LoRA configuration
        architecture = get_model_architecture(cfg.name)
        lora_config = MODEL_REGISTRY[architecture]["lora_config"]
        lora_cfg = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=cfg.lora_modules,
            lora_dropout=lora_config["dropout"],
            bias=lora_config["bias"],
            task_type=TaskType.SEQ_CLS
        )
        
        # Handle classifier head
        for attr in ["score", "classifier"]:
            if hasattr(model, attr):
                setattr(
                    model,
                    attr,
                    AutoCastingScore(getattr(model, attr), output_dtype=model.dtype),
                )
                break
        else:
            raise ValueError("Could not find classifier head in model.")
            
        model = get_peft_model(model, lora_cfg)
    
    # Convert trainable parameters to float32
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
            
    return model

def init_model_and_tokenizer(cfg: ModelConfig):
    tokenizer = init_tokenizer(cfg)
    model = init_model(tokenizer, cfg)
    return model, tokenizer

# Updated model registry with architecture-specific configs
MODEL_REGISTRY = {
    "gpt2": {
        "lr": 5e-4,
        "lora_modules": GPT2_LORA_MODULES,
        "lora_config": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.1,
            "bias": "none"
        }
    },
    "gpt2-large": {
        "lr": 3e-4,
        "lora_modules": GPT2_LORA_MODULES,
        "lora_config": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "bias": "none"
        }
    },
    "gpt2-xl": {
        "lr": 2e-4,
        "lora_modules": GPT2_LORA_MODULES,
        "lora_config": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.1,
            "bias": "none"
        }
    },
    "llama": {
        "lr": 8e-5,
        "lora_modules": LLAMA_LORA_MODULES,
        "lora_config": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "bias": "none"
        }
    },
    "mistral": {
        "lr": 8e-5,
        "lora_modules": LLAMA_LORA_MODULES,
        "lora_config": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "bias": "none"
        }
    },
    "qwen": {
        "lr": 5e-4,
        "lora_modules": LLAMA_LORA_MODULES,
        "lora_config": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.1,
            "bias": "none"
        }
    },
    "gemma": {
        "lr": 8e-5,
        "lora_modules": LLAMA_LORA_MODULES,
        "lora_config": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "bias": "none"
        }
    }
}