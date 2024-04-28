import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(r"/home/wstigall/workspace/")
from collections import Counter
import transformers
from transformers import LlamaTokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig, pipeline,AutoTokenizer
import torch
import pandas as pd
import gc

from HKLLM.hkllm.promptlm.utils.data import append_to_json_file, prepare_dataset_for_inference,extract_generated_text, rlhf_sample, create_instruction_pair, update_indices_csv
from HKLLM.hkllm.promptlm.utils.metrics import sample_recall, sample_accuracy, sample_f1_score, sample_precision, calculate_binary_metrics, count_distribution
from HKLLM.hkllm.promptlm.utils.parsers import multc_parser, parse_output_for_answer


from HKLLM.secrets import import_knowledge_file, import_class_definitions,import_observed_definitions

def inference_model(model_name,lora_adapter=None,system_prompt="",user_prompt="",follow_up="",override=False,include_supp=False):
    
        model_name = model_name
        
        data = r"/home/wstigall/workspace/300_gt_deided_case.csv"
        df = pd.read_csv(data)



        class_def = import_class_definitions()
        if include_supp == True:
            dataset = prepare_dataset_for_inference(df=df,text_col="PublicNarrative",class_col="BHR_type",sample_size=297,supp_columns=["Supp","OfficerNarrative"])
        elif include_supp == False:
            dataset = prepare_dataset_for_inference(df=df,text_col="PublicNarrative",class_col="BHR_type",sample_size=297,supp_columns=None)
            
       

       
        
        
        
            

        
        
        
        system_prompt = system_prompt
        user_prompt = user_prompt
        follow_up = follow_up
                
        texts = dataset["x"]
        labels = dataset["y"]
        
        extracted_answers = []
        compute_dtype = getattr(torch, "bfloat16")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=None,
            attn_implementation = 'flash_attention_2',
            
            low_cpu_mem_usage=True
)
        if lora_adapter:
            lora_adapter_name = lora_adapter
            model.load_adapter(lora_adapter_name)
            
        pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=model_name,
    token="",
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
        
    },
)
        
        
        for i, text in enumerate(texts):
            messages = []
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


            prompt = pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                    
        )

            

            system_prompt = system_prompt
                    
            
        
            
            
            
                
            
            sequences = pipeline(
            
            prompt,
            
            do_sample=True,
            max_new_tokens = 600,
            temperature = 0.60,
            top_p = .90,
            num_return_sequences = 1,
            
            
            
            
        )
            model_output = sequences[0]['generated_text']
            print(sequences[0]['generated_text'])
            keywords = [
                "Domestic Social", "Domestic_Social", "Mental Health", "Mental_Health", "Substance Abuse", "Substance_Abuse",
                "NonDomestic_Social", "NonDomestic Social", "Other", "SubstanceAbuse,DomesticSocial,NonDomesticSocial",
                "MENTAL_HEALTH", "DOMESTIC_SOCIAL", "NONDOMESTIC_SOCIAL", "OTHER", "SUBSTANCE_ABUSE"
            ]

            # Extracting the answer using some function
            extracted_answer = parse_output_for_answer(model_output, keywords=keywords, single_output=True)
            print(extracted_answer)

            # Handling output if it's a list, assuming single_output means a list of one item
            extracted_answer = extracted_answer[0] if isinstance(extracted_answer, list) and extracted_answer else None

            # Mapping deviations to standardized outputs
            standardized_answers = {
                'SubstanceAbuse': 'Substance_Abuse',
                'SUBSTANCE_ABUSE': 'Substance_Abuse',
                'Substance abuse': 'Substance_Abuse',
                'Substance_Abuse': 'Substance_Abuse',
                'SUBSTANCE ABUSE': 'Substance_Abuse',
                'DomesticSocial': 'Domestic_Social',
                'Domestic social': 'Domestic_Social',
                'Domestic_Social': 'Domestic_Social',
                'DOMESTIC_SOCIAL': "Domestic_Social",
                'DOMESTIC SOCIAL': "Domestic_Social",
                'NonDomesticSocial': 'NonDomestic_Social',
                'NonDomestic Social': 'NonDomestic_Social',
                'NonDomestic_Social': 'NonDomestic_Social',
                'NONDOMESTIC_SOCIAL': "NonDomestic_Social",
                'NONDOMESTIC SOCIAL': "NonDomestic_Social",
                'MentalHealth': 'Mental_Health',
                'Mental health': 'Mental_Health',
                'Mental_Health': 'Mental_Health',
                'MENTAL_HEALTH': 'Mental_Health',
                'MENTAL HEALTH': 'Mental_Health',
                'Other': 'Other',  # Ensuring 'Other' is directly mapped
                'OTHER': 'Other',
                'other': "Other"
            }


            # Others category consolidation
            others_category = [None, "", "Others", "Other_", "Others_"]

            # Process extracted answer
            if extracted_answer in others_category or extracted_answer is None:
                processed_answer = 'Other'
                if override:
                    processed_answer = input("Answer was not correctly processed, Enter the correct Tag: ").replace(" ", "_")
                extracted_answers.append(processed_answer)
            elif extracted_answer in standardized_answers:
                processed_answer = standardized_answers[extracted_answer]
                extracted_answers.append(processed_answer)
            else:
                # Handle the default case where answer does not need modification except for replacing spaces with underscores
                processed_answer = extracted_answer.replace(" ", "_") if isinstance(extracted_answer, str) else None
                if processed_answer:
                    extracted_answers.append(processed_answer)
                
            is_correct = "Correct" if processed_answer == labels[i] else "Incorrect"
            print(f"Sample {i}: Extracted Answer = {processed_answer}, Label = {labels[i]}, {is_correct}")
            print(len(extracted_answers),len(labels))
            gc.collect()
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
                
user_prompt = """[INST]As the police department's premier virtual assistant, your task is to analyze the following narratives to determine current or future behavioral health indicators, independent of any danger or threat. Each narrative should be tagged with only one of the following Behavioral Health subcategories: Domestic Social, NonDomestic Social, Mental Health, Substance Abuse, or Other. Note that there are no boundaries on predicting potential issues; any relevant signals should be appropriately classified. Begin your response using the format: <Tag> [Answer] </Tag>. Proceed with the text classification: [/INST]\n[INST]
"""
follow_up = """[/INST][INST]End of Text. Please classify the text using the specified format starting with <Tag>Answer</Tag>. Include intermediate reasoning to support your classification decision. Please output the tag first: [/INST]"""

system_prompt = """[INST]Behvaioral Health Classification Overview:\n
                Behavioral Health involves prompting mental health, resilience, and well-being. it includes the treatment of mental and substance abuse disorders,adressing domestic disputes in communities, as well as providing support to affected individuals and their families. This classification aims at early detection of behavioral health issues to facilitate timely intervention and better managment, focusing on any level of risk rather than only clear and present dangers\n
                Categories of Behavioral Health:\n
                1.Mental Health\n
                ##Definiiton: Involves individuals diagonsed or suspected of having a mental disorder such as schizophrenia or exhibiting suicidal ideations\n
                ##Crisis Situations: Includes cases of mental health crises like anxiety attacks, paranoia and anger issues\n
                ##Potential Keywords  "acting strange",   "ADHD",   "altered",   "alzheimer",   "anger management",   "antisocial",   "anxiety",   "asperger",   "autisti",   "behavioral disorder",   "bi-polar",   "bipolar",   "bizarre",   "conspiracy",   "crazy",   "delusional",   "demented",   "dementia",   "depressed",   "depression",   "developmentally challenged",   "emotionally disturbed",   "Erratic",   "free masons",   "hallucinat",   "irrational",   "illuminati",   "manic",   "mental",   "mental illness",   "mind",   "overdos","paranoid",   "paranoia",   "psychotic",   "psychiatric",   "Ridgeview",   "schizophrenia",   "schizoaffective",   "self harm",   "spectrum",   "spirits",   "strange",   "Suicid",   "syndrome",   "therap",   "weird" make sure to consider the contex\n
                2.Domestic Social:\n
                ##Definition: Pertains to conflicts within domestic settings, involving relationships like spouses, girlfriends, or parent-child interactions, this extends to any situations where the involved peoples share the same living space.\n
                ##Examples: Includes domestic disputes, child custody disputes, and other related incidient, not restricted to physical home settings, extends to neighbors and community of there is a domestic relationship between the involved peoples.\n
                ##Common Keywords: "child",   "adolescent",   "juvenile",   "teen",   "youth", "toddler",   "infant",   "baby",   "kid", "abduction",   "abrasion",   "abuse","altercation",   "amber alert",      "argu",     "attack",   "banging",   "belligerent",   "berate",    "bother",   "bruis",   "breath",     "chok",     "confuse",   "crying",   "custody",   "cursing",   "cut",   "cuts",  "dispute",   "Domestic",   "domestic violence", "elderly missing",   "escape",   "estranged",     "evict",  "fight",      "harass",   "harm",      "injury", "protective order",   "punch",   "push",   "refusing",   "runaway",   "scared",   "scream",   "shoved",   "spat",   "strangle",   "strangulate",   "struck",   "struggle",   "suspicious",   "surveil",   "swing",   "swollen",   "threat",   "threaten",   "traumatic",      "uncooperative","wrestl",   "yelled",   "yelling"
                3.NonDomestic Social\n
                ##Definition: Involves conflicts between individuals outside of a domestic setting.\n
                ##Notes: Typically includes cases where crimes are commited against individuals not personally related to the perpretrator, this is a comparatively rare classification compared to Other\n
                4.Substance Abuse\n
                ##Definition: Focuses on individuals with ongoing drug or alcohol abuse issues.\n
                ##Identification: Evidence of abuse must be present, beyond mere possession of controlled substances. For instance, the presence of drug paraphernalia or symptoms of intocication might indicate an abuse issue, wheras the mere odor of marijuana wihtout other evidence does not qualify.\n
                
                5.Other\n
                ##Definition: this encompassess all Non-Behavioral Health samples, these samples are referred to as Other, things that will always fall into this category are Animal Attacks, Shoplifting, Breaking in Entering, Misdemeanor or lower traffic infractions, and Other petty thefts.\n
                ##Other Considerations: This should also include all indidences when a police report has been made, and there are no individuals mentioned in the report.\n
                Subcategory Precedence:\n
                ##If a person commits suicide via drug overdose it shall be classified as Mental Health, not Substance Abuse\n
                ##If somebody attacks a member of their family due to a pre-diagnosed mental health disorder like schizophrenia it shall be classified as Mental Health, not Domestic Social.\n
                ##If somebody breaks into something it shall be classified as Other instead of NonDomestic Social\n
                ##If a person shoplifts it shall be classified as Other instead of NonDomestic Social\n
                ##Unattended substances cannot be classified as Substance Abuse and should be classified as Other\n
                ##There are no such thing as "crime" and "none" tags these should be considered "Other"\n
                Objective of Classification:\n
                The Primary goal of this classification system is not only to manage existing crises but also to identify potential future behavioral health crises, thereby enablign earlier and more effective interventions[/INST]
                """
lora_adapter = r"/home/wstigall/workspace/Llama-3-HELPR"
#inference_model(lora_adapter=lora_adapter,model_name="mistralai/Mistral-7B-Instruct-v0.2",system_prompt=system_prompt,use_class_definitions=True,override=True,include_supp=True)
inference_model(model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",lora_adapter=lora_adapter,system_prompt=system_prompt,follow_up=follow_up,user_prompt=user_prompt,include_supp=True,override=True)
#inference_model(lora_adapter=None,model_name="mistralai/Mistral-7B-Instruct-v0.2",use_class_definitions=True,include_supp=True,override=True)