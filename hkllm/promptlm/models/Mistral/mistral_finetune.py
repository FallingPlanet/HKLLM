import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from datasets import load_dataset,concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

from trl import SFTTrainer
from torch.utils.data import Dataset,DataLoader
import json
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

# Model from Hugging Face hub
base_model = "mistralai/Mistral-7B-Instruct-v0.2"

class MyDataset(Dataset):
  def __init__(self, json_file):
    # read the json file and store it as a Python object
    with open(json_file, 'r') as f:
      self.data = json.load(f)
  
  def __len__(self):
    # return the number of samples in the data
    return len(self.data)
  
  def __getitem__(self, idx):
    # get the input and target from the data
    input = self.data[idx]['prompt']
    target = self.data[idx]['completion']
    # return a tuple of (input, target)
    return (input, target)

# New instruction dataset
#data_files = r"/home/wstigall/workspace/lora_dataset.json"
data_files = r"/home/wstigall/workspace/merged_synthetic_lora.json"
#data_files2 = r"/home/wstigall/workspace/lora_dataset.json"
data_files3 = r"/home/wstigall/workspace/synthetic_lora2.json"

eval_data_files = r"/home/wstigall/workspace/lora_silver_dataset.json"
eval_data_files2 = r"/home/wstigall/workspace/new_synthetic_lora.json"
#dataset2 = load_dataset('json',data_files=data_files2,split="train")
dataset = load_dataset('json',data_files=data_files,split="train")
dataset3 = load_dataset('json',data_files=data_files3,split="train")
eval_dataset = load_dataset('json',data_files=eval_data_files,split="train")
eval_dataset2 = load_dataset('json',data_files=eval_data_files2,split="train")




merged_dataset = concatenate_datasets([dataset,dataset3])
eval_merged = concatenate_datasets([eval_dataset,eval_dataset2])
print(len(merged_dataset))





# Fine-tuned model
new_model = "Mistral-HELPR-v4"

compute_dtype = getattr(torch, "bfloat16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=None,
    attn_implementation = 'flash_attention_2'
)
model.config.use_cache = False
model.config.pretraining_tp = 1

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

training_params = TrainingArguments(
    output_dir="./mistral_HELPR/results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=5,
    learning_rate=2e-6,
    weight_decay=0.015,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant":False}
    
    
)



trainer = SFTTrainer(
    model=model,
    train_dataset=merged_dataset,
    eval_dataset=eval_merged,
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=2500,

    tokenizer=tokenizer,
    args=training_params,
    packing=False,
    neftune_noise_alpha=10
    
)

trainer.train()

trainer.model.save_pretrained(new_model)
model.config.use_cache=True