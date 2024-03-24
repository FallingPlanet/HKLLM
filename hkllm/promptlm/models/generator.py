import sys
import os

sys.path.append(r"/home/wstigall/workspace/")
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import pandas as pd

from HKLLM.hkllm.promptlm.utils.data import append_to_json_file, prepare_dataset_for_generator,extract_generated_text, rlhf_sample, create_instruction_pair, update_indices_csv
from HKLLM.hkllm.promptlm.utils.metrics import sample_recall, sample_accuracy, sample_f1_score, sample_precision
from HKLLM.hkllm.promptlm.utils.parsers import multc_parser, parse_output_for_answer

from peft import PeftModel, PeftConfig, AutoPeftModel, AutoPeftModelForCausalLM
from HKLLM.secrets import import_knowledge_file, import_class_definitions

#Change to mistralai/Mixtral-8x7B-v0.1
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

#Change the path of the dataset in your workspace
data = r"//home/wstigall/pain/modified_deid_200k_main.csv"
df = pd.read_csv(data,low_memory=False)
df['index'] = range(1, len(df)+1)

#Change the path of the repository in your workspace
indicies_path = r"/home/wstigall/workspace/HKLLM/parsed_indicies.csv"
df.set_index('index', inplace=True)

#Change the path of the repository in your workspace
indicies_path = r"/home/wstigall/workspace/HKLLM/parsed_indicies.csv"
dataset = prepare_dataset_for_generator(indices_column_name='Index',indices_csv_path=indicies_path,df=df,text_col="narrative1",class_col=None,sample_size=1000,supp_columns=None)
quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype = torch.bfloat16
    )

#REMOVE THESE LINES
lora_adapter_name = "/home/wstigall/workspace/mistral-lora-v0.4"
model.load_adapter(lora_adapter_name)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    device_map="auto",
    torch_dtype=torch.bfloat16
    
)




system_prompt = """[INST]You are the police department's Ace virtual assistant, you are going to read the following narratives
        and return whether they are related to behavioral health, All samples only have one answer.  
        For your response you must always use <Tag> [Answer] </Tag>.
        you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse or Other there are no other tags.
        The text you must classify is as follows: [/INST]"""
        
texts = dataset["x"]
indicies = dataset["Index"]

class_def = import_class_definitions()

full_prompts = [str(class_def)+str(system_prompt)+str(text)+"[INST]  Classify the text, you must ONLY use <Tag> [Answer] </Tag> and can choose ONLY one answer, give in-depth intermediate reasoning[/INST]" for text in texts ]
n=0
extracted_answers = []
for prompt in full_prompts:

    sample_index = indicies[n]
    n+=1
    sequences = pipe(
            prompt,
            do_sample=True,
            max_new_tokens = 1000,
            temperature = 0.50,
            top_k=100,
            top_p = 0.99,
            num_return_sequences = 2                                                                                    ,
            repetition_penalty = 1.15
        )
    model_output_one = sequences[0]['generated_text']
    model_output_two = sequences[1]['generated_text']
    keywords = ["Domestic Social","Domestic_Social","Mental Health","Mental_Health","Substance Abuse","Substance_Abuse","NonDomestic_Social","NonDomestic Social","Other","SubstanceAbuse"]
    extracted_answer = (parse_output_for_answer(model_output_one,keywords=keywords,single_output=True))
    extracted_answer_two = (parse_output_for_answer(model_output_two,keywords=keywords,single_output=True))
    print(sequences[0]['generated_text'])
    print(extracted_answer)
    print(sequences[1]['generated_text'])
    print(extracted_answer_two)
    
    generated_sequence_1 = extract_generated_text(model_output_one,prompt)
    print(f"Generated Answer 1:{generated_sequence_1}")
    generated_sequence_2 = extract_generated_text(model_output_two,prompt)
    print(f"Generated Answer 2: {generated_sequence_2}")
    def is_valid_input(input):
        if input not in [1,2,3,4,5,6,7]:
            return False
        else: return True
        
    selected_sequence = input("Select the best sample from the prompts\nPress 1 to select the first prompt\nPress 2 to select the second prompt\nPress 3 if neither prompt for both lora and dpo\nPress 4 to add the first prompt to Lora ONLY\nPress 5 to add the second prompt to Lora ONLY\nPress 6 to add the first prompt to DPO ONLY \nPress 7 to add the second prompt to DPO only \n")
    while not is_valid_input(input=int(selected_sequence)):
        selected_sequence = input("Invalid Input, Select the best sample from the prompts\nPress 1 to select the first prompt\nPress 2 to select the second prompt\nPress 3 if neither prompt for both lora and dpo\nPress 4 to add the first prompt to Lora ONLY\nPress 5 to add the second prompt to Lora ONLY\nPress 6 to add the first prompt to DPO ONLY \nPress 7 to add the second prompt to DPO only \n")
    lora_dataset = "/home/wstigall/workspace/lora_eval_dataset.json"
    dpo_dataset = "/home/wstigall/workspace/filtered_dpo.json"
    update_indices_csv(csv_path=r"/home/wstigall/workspace/HKLLM/parsed_indicies.csv",indices_column_name="Index",new_index=sample_index)
    if selected_sequence in [1,"1"]:
        dpo_sample = rlhf_sample(prompt=prompt,accepted=generated_sequence_1,rejected=generated_sequence_2)
        lora_sample = create_instruction_pair(prompt=prompt,completion=generated_sequence_1)
        append_to_json_file(dpo_dataset,new_data=dpo_sample)
        append_to_json_file(lora_dataset,lora_sample)
        
    if selected_sequence in [2,'2']:
        dpo_sample = rlhf_sample(prompt=prompt,accepted=generated_sequence_2,rejected=generated_sequence_1)
        lora_sample = create_instruction_pair(prompt=prompt,completion=generated_sequence_2)
        append_to_json_file(dpo_dataset,new_data=dpo_sample)
        append_to_json_file(lora_dataset,lora_sample)
        
    if selected_sequence in [3,'3']:
        dpo_sample = rlhf_sample(prompt=prompt,accepted="[NULL REASONING]",rejected=generated_sequence_1)
        lora_sample = create_instruction_pair(prompt=prompt,completion="[NULL REASONING]")
        append_to_json_file(dpo_dataset,new_data=dpo_sample)
        append_to_json_file(lora_dataset,lora_sample)
    if selected_sequence in [4,'4']:
        lora_sample=create_instruction_pair(prompt=prompt,completion=generated_sequence_1)
        append_to_json_file(lora_dataset,lora_sample)
    if selected_sequence in [5,'5']:
        lora_sample = create_instruction_pair(prompt=prompt,completion=generated_sequence_2)
        append_to_json_file(lora_dataset,lora_sample)
    if selected_sequence in [6,'6']:
        dpo_sample = rlhf_sample(prompt=prompt,accepted=generated_sequence_1,rejected=generated_sequence_2)
        append_to_json_file(dpo_dataset,new_data=dpo_sample)
    if selected_sequence in [7,'7']:
        dpo_sample = rlhf_sample(prompt=prompt,accepted=generated_sequence_2,rejected=generated_sequence_1)
        append_to_json_file(dpo_dataset,new_data=dpo_sample)
    
    
        
    
        
        
        
    