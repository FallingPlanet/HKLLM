import sys
import os

sys.path.append(r"/home/wstigall/workspace/")
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, LlamaTokenizerFast
import torch
import pandas as pd

from HKLLM.hkllm.promptlm.utils.data import append_to_json_file, prepare_dataset_for_generator,extract_generated_text, rlhf_sample, create_instruction_pair, update_indices_csv
from HKLLM.hkllm.promptlm.utils.metrics import sample_recall, sample_accuracy, sample_f1_score, sample_precision
from HKLLM.hkllm.promptlm.utils.parsers import multc_parser, parse_output_for_answer

from peft import PeftModel, PeftConfig, AutoPeftModel, AutoPeftModelForCausalLM
from HKLLM.secrets import import_knowledge_file, import_class_definitions


#model_name = "mistralai/Mixtral-8x7B-v0.1"

data = r"//home/wstigall/pain/modified_deid_200k_main.csv"
df = pd.read_csv(data,low_memory=False)
df['index'] = range(1, len(df)+1)
indicies_path = r"/home/wstigall/workspace/HKLLM/parsed_indicies.csv"
df.set_index('index', inplace=True)


indicies_path = r"/home/wstigall/workspace/HKLLM/parsed_indicies.csv"
dataset = prepare_dataset_for_generator(indices_column_name='Index',indices_csv_path=indicies_path,df=df,text_col="narrative1",class_col=None,sample_size=1000,supp_columns=None)
quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            
        )

lora_adapter= None
model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        use_auth_token = "",
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype = torch.bfloat16
    )
if lora_adapter:
    lora_adapter_name = lora_adapter
    model.load_adapter(lora_adapter_name)
    

tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)

user_prompt = """[INST]You are the police department's Ace virtual assistant, you are going to read the following narratives
                and return whether they are current or future indicators of behavioral health regardless of indication of danger or threat, All samples only have one tag it is impossible to select more than one tag for a given narrative.  
                For your response you must always use <Tag> [Answer] </Tag>.
                you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse, or Other;
                It should be considered that there is not a prediction boundary, and any signals should be classified as such 
                The text you must classify is as follows: [/INST][INST]"""
follow_up = "[/INST][INST]End of Text. Classify the text, you are restricted to beginning with <Tag>Answer</Tag> Give intermediate reasoning behind your classification of the report[/INST]"
system_prompt = """Behavioral Health: Can only be applied to people, The definition of behavioral health in this context is The promotion of mental health, resilence and wellbeing, the treatment of mental and substance use disorders, and support for these peoples along with their families. Behavioral health includes current behavioral health crisis as well as marking cases that could be considered as having the potential to cause a future behavioral health crisis,The goal of Behavioral Health Classification is early detection and finding signs that may lead to early treatment and better handling of effected individuals therefore there is not the need to assess clear and present danger but any level of risk can warrant classification. The classes of Behavioral Health is as follows:\nMental Health: Involving an individual with a diagnosed mental disorder, like schizophrenia or sucidal ideations, however this can include individuals that are having a mental health crisis, these mental health crisis can be anxiety attacks, paranoia, anger issues,clearly unusual behaviors, or other materializations of mental health incidents\nDomestic Social: Involving multiple Individuals in a home setting, like husband/wife or parent/children domestic disputes,and all other kinds of domestic disputes and incidents including child custody disputes, It is not necessary for it to be in a home setting if the relationship between the people is domestic.Events involving conflicts between people, with the potential to reoccur are generally considered Domestic Social regardless of whether the parties involved came to an agreement.\nNonDomestic Social: Involving multiple individuals not in a home setting, like comitting crimes on those not related to the perpretrator (This is rare by the way)\nSubstance Abuse: Individual with persistent drug/alchohol abuse problems, Posession alone is not enough to indicate abuse problems. If an officer pulls somebody over for a traffic stop and they are not under the influence but they are in possession of controlled substances it is not substance abuse, so look for evidence of abuse, If the only evidence is the oder of marijuana then it is Other"""
system_prompt = system_prompt
user_prompt = user_prompt
follow_up = follow_up
        
texts = dataset["x"]
indicies = dataset["Index"]


pipe = pipeline(
"text-generation",
model=model,
tokenizer=tokenizer,
trust_remote_code=True,

)
messages = []
for i, text in enumerate(texts):
    messages.append({
        "role": "system", 
        "content": system_prompt
    })
    messages.append({
        "role": "user", 
        "content": user_prompt
    })
    messages.append({
        "role": "user", 
        "content": text
    })
    messages.append({
        "role": "user",
        "content": follow_up
    })


    prompt = pipe.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
)

        

system_prompt = system_prompt
        



extracted_answers = []
        
        
            
        
sequences = pipe(

prompt,

do_sample=True,
max_new_tokens = 150,
temperature = 0.25,
top_k=1,
top_p = .10,
num_return_sequences = 1,
repetition_penalty = 1.5

        
        
    )
model_output_one = sequences[0]['generated_text']
model_output_two = sequences[1]['generated_text']
keywords = [
            "Domestic Social", "Domestic_Social", "Mental Health", "Mental_Health", "Substance Abuse", "Substance_Abuse",
            "NonDomestic_Social", "NonDomestic Social", "Other", "SubstanceAbuse,DomesticSocial,NonDomesticSocial",
            "MENTAL_HEALTH", "DOMESTIC_SOCIAL", "NONDOMESTIC_SOCIAL", "OTHER", "SUBSTANCE_ABUSE"
        ]
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
    dpo_sample = rlhf_sample(prompt=prompt,chosen=generated_sequence_1,rejected=generated_sequence_2)
    lora_sample = create_instruction_pair(prompt=prompt,completion=generated_sequence_1)
    append_to_json_file(dpo_dataset,new_data=dpo_sample)
    append_to_json_file(lora_dataset,lora_sample)
    
if selected_sequence in [2,'2']:
    dpo_sample = rlhf_sample(prompt=prompt,chosen=generated_sequence_2,rejected=generated_sequence_1)
    lora_sample = create_instruction_pair(prompt=prompt,completion=generated_sequence_2)
    append_to_json_file(dpo_dataset,new_data=dpo_sample)
    append_to_json_file(lora_dataset,lora_sample)
    
if selected_sequence in [3,'3']:
    dpo_sample = rlhf_sample(prompt=prompt,chosen="[NULL REASONING]",rejected=generated_sequence_1)
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
    dpo_sample = rlhf_sample(prompt=prompt,chosen=generated_sequence_1,rejected=generated_sequence_2)
    append_to_json_file(dpo_dataset,new_data=dpo_sample)
if selected_sequence in [7,'7']:
    dpo_sample = rlhf_sample(prompt=prompt,chosen=generated_sequence_2,rejected=generated_sequence_1)
    append_to_json_file(dpo_dataset,new_data=dpo_sample)