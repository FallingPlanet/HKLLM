import os

import torch
from datasets import load_dataset,concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    LlamaTokenizerFast
)

from trl import SFTTrainer
from torch.utils.data import Dataset,DataLoader
import json
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

# Model from Hugging Face hub
base_model = "unsloth/llama-3-8b-Instruct-bnb-4bit"

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
data_files3 = r"/home/wstigall/workspace/new_synthetic_lora.json"

eval_data_files = r"/home/wstigall/workspace/new_data_instruct.json"
#dataset2 = load_dataset('json',data_files=data_files2,split="train")
dataset = load_dataset('json',data_files=data_files,split="train")
dataset3 = load_dataset('json',data_files=data_files3,split="train")
eval_dataset = load_dataset('json',data_files=eval_data_files,split="train")






merged_dataset = concatenate_datasets([dataset,dataset3])
print(len(merged_dataset))



def instruction_to_messages(dialog_entries):
    # Converts dialog entries from instruction format to messages format
    messages_entries = []
    for entry in dialog_entries:
        messages = [
            {"role": "user", "content": entry["prompt"]},  # Assuming the prompt is user's input
            {"role": "assistant", "content": entry["completion"]}  # Assuming the completion is assistant's response
        ]
        messages_entries.append({"messages": messages})
    return messages_entries

def llama_3_formatting_func(dialog_entries, convert_from_instruction=True, bos_token="<|startoftext|>", eos_token="<|endoftext|>"):
    if convert_from_instruction:
        dialog_entries = instruction_to_messages(dialog_entries)

    formatted_entries = []

    for entry in dialog_entries:
        formatted_text = bos_token
        for message in entry["messages"]:
            # Format each message by role
            formatted_text += f"{message['role']}\n\n{message['content'].strip()}\n"
        formatted_text += eos_token  # Append EOS token to signify end of the input
        formatted_entries.append(formatted_text)

    return formatted_entries

# Example usage with initial instruction format data
instruction_entries = [
    {"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."},
    {"prompt": "Who wrote 'Romeo and Juliet'?", "completion": "William Shakespeare wrote 'Romeo and Juliet'."}
]

# Apply LLaMA-3 formatting with conversion from instruction format
formatted_entries = llama_3_formatting_func(instruction_entries, convert_from_instruction=True)
for formatted_entry in formatted_entries:
    print("Formatted Entry:", formatted_entry)



# Fine-tuned model
new_model = "Llama-3-HELPR"

compute_dtype = getattr(torch, "bfloat16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=None,
    attn_implementation = 'flash_attention_2',
    
    low_cpu_mem_usage=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit", trust_remote_code=True,use_fast = False)
tokenizer.padding_side = "right"



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
    output_dir="./llama_256Lora3/results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=5,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=5,
    learning_rate=2e-7,
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
    tokenizer=tokenizer,
    formatting_func=llama_3_formatting_func,
    train_dataset=merged_dataset,
    eval_dataset=None,
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=1200,

    
    args=training_params,
    packing=False,
    neftune_noise_alpha=10
    
)

trainer.train()

trainer.model.save_pretrained(new_model)
model.config.use_cache=True