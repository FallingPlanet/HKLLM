import os
import sys
sys.path.append(r"/home/wstigall/workspace/")
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import pandas as pd

from HKLLM.hkllm.promptlm.utils.data import append_to_json_file, prepare_dataset_for_inference,extract_generated_text, rlhf_sample, create_instruction_pair, update_indices_csv
from HKLLM.hkllm.promptlm.utils.metrics import sample_recall, sample_accuracy, sample_f1_score, sample_precision, calculate_binary_metrics, count_distribution
from HKLLM.hkllm.promptlm.utils.parsers import multc_parser, parse_output_for_answer

from peft import PeftModel, PeftConfig, AutoPeftModel, AutoPeftModelForCausalLM
from HKLLM.secrets import import_knowledge_file, import_class_definitions
def inference_model(model_name,lora_adapter=None,system_prompt="",use_class_definitions=False,use_knowledge_files=False,override=False):
    
        model_name = model_name

        data = r"/home/wstigall/workspace/300_gt_deided_case.csv"
        df = pd.read_csv(data)



        class_def = import_class_definitions()

        dataset = prepare_dataset_for_inference(df=df,text_col="PublicNarrative",class_col="BHR_type",sample_size=297,supp_columns=["Supp","OfficerNarrative"])

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
        for i, prompt in enumerate(full_prompts):
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
            
            
            if extracted_answer == None or extracted_answer == [] or extracted_answer == "":
                processed_answer = 'Other'
                if override == True:
                    processed_answer = input("Answer was not correctly processed, Enter the correct Tag: ")
                    processed_answer = processed_answer.replace(" ","_")
                extracted_answers.append(processed_answer)
            elif extracted_answer == 'SubstanceAbuse':
                processed_answer = 'Substance Abuse'
                extracted_answers.append(processed_answer)
            elif extracted_answer == 'DomesticSocial':
                processed_answer = 'Domestic Social'
                extracted_answers.append(processed_answer)
            elif extracted_answer == "NonDomesticSocial":
                processed_answer = "NonDomestic Social"
                extracted_answers.append(processed_answer)
            else:
                processed_answer = extracted_answer[0].replace(" ","_")
                extracted_answers.append(processed_answer)
                
            is_correct = "Correct" if processed_answer == labels[i] else "Incorrect"
            print(f"Sample {i}: Extracted Answer = {processed_answer}, Label = {labels[i]}, {is_correct}")
            print(len(extracted_answers),len(labels))
        accuracy = sample_accuracy(y_true=labels,y_pred=extracted_answers)
        precision = sample_precision(y_true=labels,y_pred=extracted_answers,macro=True)
        recall = sample_recall(y_true=labels,y_pred=extracted_answers,macro=True)
        f1_score = sample_f1_score(y_true=labels,y_pred=extracted_answers,macro=True)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"f1: {f1_score:.2f}")
        positive_classes = ["Substance_Abuse", "Domestic_Social", "NonDomestic_Social", "Mental_Health"]  # Example subclass labels
        b_accuracy, b_precision, b_recall, b_f1 = calculate_binary_metrics(y_true=labels, y_pred=extracted_answers, positive_classes=positive_classes)

        print(f"Binary Accuracy: {b_accuracy}")
        print(f"Binary Precision: {b_precision}")
        print(f"Binary Recall: {b_recall}")
        print(f"Binary F1 Score: {b_f1}")

        actual_label_distributions,pred_label_distribution = count_distribution(y_true=labels,y_pred=extracted_answers)
        print(actual_label_distributions,pred_label_distribution)
                
system_prompt = """[INST]You are the police department's Ace virtual assistant, you are going to read the following narratives
                and return whether they are related to behavioral health, All samples only have one answer.  The classification of the sample
                is based on the current report. For your response you must always use <Tag> [Answer] </Tag>.
                you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse or Other there are no other tags.
                The text you must classify is as follows: [/INST]"""
lora_adapter = r"/home/wstigall/workspace/mistral-hkllm-ksu-internal-v0.9"
inference_model(lora_adapter=None,model_name="mistralai/Mistral-7B-Instruct-v0.2",system_prompt=system_prompt,use_class_definitions=True,override=True)
#inference_model(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",lora_adapter=None,system_prompt=system_prompt)
#inference_model(lora_adapter=None,model_name="mistralai/Mistral-7B-Instruct-v0.2")