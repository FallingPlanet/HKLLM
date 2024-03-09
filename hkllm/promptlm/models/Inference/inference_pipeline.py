import os
import sys
sys.path.append(r"/home/wstigall/workspace/")
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import pandas as pd

from HKLLM.hkllm.promptlm.utils.data import append_to_json_file, prepare_dataset_for_inference,extract_generated_text, rlhf_sample, create_instruction_pair, update_indices_csv
from HKLLM.hkllm.promptlm.utils.metrics import sample_recall, sample_accuracy, sample_f1_score, sample_precision,calculate_binary_metrics 
from HKLLM.hkllm.promptlm.utils.parsers import multc_parser, parse_output_for_answer

from peft import PeftModel, PeftConfig, AutoPeftModel, AutoPeftModelForCausalLM
from HKLLM.secrets import import_knowledge_file, import_class_definitions
def inference_model(model_name,lora_adapter=None,system_prompt="",use_class_definitions=False,use_knowledge_files=False):
    
        model_name = model_name

        data = r"/home/wstigall/workspace/300_gt_deided_case.csv"
        df = pd.read_csv(data)



        class_def = import_class_definitions()

        dataset = prepare_dataset_for_inference(df=df,text_col="PublicNarrative",class_col="BHR_type",sample_size=297,supp_columns=None)

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
        if lora_adapter:
            lora_adapter_name = lora_adapter
            model.load_adapter(lora_adapter_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer = tokenizer, 
            device_map="auto",
            torch_dtype=torch.bfloat16
            
        )

        system_prompt = system_prompt
                
        texts = dataset["x"]
        labels = dataset["y"]
        print(len(texts))
        print(len(labels))
        if use_class_definitions:
            full_prompts = [str(class_def)+str(system_prompt)+str(text)+"[INST]  Classify the text, you must ONLY use <Tag> [Answer] </Tag> and can choose ONLY one answer, give intermediate reasoning[/INST]" for text in texts ]
        else:
            full_prompts = [str(system_prompt)+str(text)+"[INST]  Classify the text, you must ONLY use <Tag> [Answer] </Tag> and can choose ONLY one answer, give intermediate reasoning[/INST]" for text in texts ]
        extracted_answers = []
        extracted_answers = []
        for prompt in full_prompts:
            len(full_prompts)
        
            sequences = pipe(
                    prompt,
                    do_sample=True,
                    max_new_tokens = 500,
                    temperature = 0.95,
                    top_k=150,
                    top_p = 0.9,
                    num_return_sequences = 1                                                                                    ,
                    
                )
            model_output = sequences[0]['generated_text']
            print(sequences[0]['generated_text'])
            keywords = ["Domestic Social","Domestic_Social","Mental Health","Mental_Health","Substance Abuse","Substance_Abuse","NonDomestic_Social","NonDomestic Social","Other","SubstanceAbuse,DomesticSocial,NonDomesticSocial"]
            extracted_answer = (parse_output_for_answer(model_output,keywords=keywords,single_output=True))
            print(extracted_answer)
            if extracted_answer == None or extracted_answer == []:
                extracted_answer = "Other"
                extracted_answers.append(extracted_answer)
            elif extracted_answer == "SubstanceAbuse":
                extracted_answer == "Substance Abuse"
                extracted_answers.append(extracted_answer)
            elif extracted_answer == "DomesticSocial":
                extracted_answer == "Domestic Social"
                extracted_answers.append(extracted_answer)
            elif extracted_answer == "NonDomesticSocial":
                extracted_answer == "NonDomestic Social"
                extracted_answers.append(extracted_answer)
            else:
                processed_answer = extracted_answer[0].replace(" ","_")
                extracted_answers.append(processed_answer)
            print(len(extracted_answers),len(labels))
        accuracy = sample_accuracy(y_true=labels,y_pred=extracted_answers)
        precision = sample_precision(y_true=labels,y_pred=extracted_answers,macro=True)
        recall = sample_recall(y_true=labels,y_pred=extracted_answers,macro=True)
        f1_score = sample_f1_score(y_true=labels,y_pred=extracted_answers,macro=True)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"f1: {f1_score:.2f}")
            

                
system_prompt = """[INST]You are the police department's Ace virtual assistant, you are going to read the following narratives
                and return whether they are related to behavioral health, All samples only have one answer.  The classification of the sample
                is based on the current report, references to past events inside of the report do not affect the classification of the report.
                For your response you must always use <Tag> [Answer] </Tag>.
                you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse or Other there are no other tags.
                You are a virtual assistant so do not refer to yourself in first person when giving reasoning. If you find no evidence of a tag the tag is Other
                There is some final context I will provide to aid you, if an animal is involved the call is Other. In this context
                if a drug is found during the traffic stop, it should not be classified as Substance Abuse unless the person was actively operating
                their vehicle under the influence. The text you must classify is as follows: [/INST]"""
inference_model(lora_adapter=r"/home/wstigall/workspace/mistral-hkllm-ksu-internal-v0.9",model_name="mistralai/Mistral-7B-Instruct-v0.2",system_prompt=system_prompt,use_class_definitions=True)
#inference_model(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",lora_adapter=None,system_prompt=system_prompt)
#inference_model(lora_adapter=None,model_name="mistralai/Mistral-7B-Instruct-v0.2")