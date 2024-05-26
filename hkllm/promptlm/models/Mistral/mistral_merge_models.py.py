import os

import torch
from datasets import load_dataset,concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

from trl import SFTTrainer, DPOTrainer

import json
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import json

import os
import accelerate

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
data_files = r"/home/wstigall/workspace/filtered_dpo.json"
data_files2 = r"/home/wstigall/workspace/1050_DPO.json"



dataset = load_dataset('json',data_files=data_files,split="train")
dataset2 = load_dataset('json',data_files=data_files2,split="train")
merged_dataset = concatenate_datasets([dataset,dataset2])
train_dataset, test_dataset = merged_dataset.train_test_split(test_size=0.15).values()
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
new_model = "Mistral-HELPR-7B-4bit"
with open('/home/wstigall/workspace/1050_DPO_fixed.json', 'r') as file:
    data = json.load(file)
compute_dtype = getattr(torch, "bfloat16")
# Calculate split sizes
num_examples = len(data['prompt'])  # Assuming 'prompt' is a key in your dictionary
train_size = int(num_examples * 0.85)

# Create training and testing datasets
train_data = {
    'prompt': data['prompt'][:train_size],
    'chosen': data['chosen'][:train_size],
    'rejected': data['rejected'][:train_size]
}

test_data = {
    'prompt': data['prompt'][train_size:],
    'chosen': data['chosen'][train_size:],
    'rejected': data['rejected'][train_size:]
}

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    token="",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
    device_map = "auto"
)
model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.unk_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token = True

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=256,
    lora_dropout=.10,
    r=256,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    
)
model = get_peft_model(model, peft_config)
model = PeftModel.from_pretrained(
    model,
    "/home/wstigall/workspace/Mistral-HELPR-v3",
    is_trainable=True,
    
)

reference = get_peft_model(model,peft_config)
reference = PeftModel.from_pretrained(
    reference,
    "/home/wstigall/workspace/Mistral-HELPR-v3",
    is_trainable=False
    
)

training_params = TrainingArguments(
    output_dir="./mistral_DPOv6/results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=5,
    learning_rate=5e-6,
    weight_decay=0.015,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant":False},
    remove_unused_columns=False,
    overwrite_output_dir=True
)
    
    



# Create DPO trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=reference,
    args=training_params,
    
    
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    
    beta=0.2,
    max_prompt_length=1500,
    
    max_length=1536,
)
# Fine-tune model with DPO
dpo_trainer.train()
dpo_trainer.save_model(new_model)