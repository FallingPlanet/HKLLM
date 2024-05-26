import torch
from unsloth import FastLanguageModel

# Set the maximum sequence length
max_seq_length = 2048

# Set the data type (None for auto-detection, Float16 for older GPUs, Bfloat16 for newer GPUs)
dtype = None

# Enable 4-bit quantization to reduce memory usage
load_in_4bit = True

# Load the pre-trained LLaMA 3 model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=256,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
from datasets import load_dataset,concatenate_datasets


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



from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
    ),
)

trainer.train()