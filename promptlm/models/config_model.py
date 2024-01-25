from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LlamaForCausalLM, LlamaTokenizerFast
)
import torch

class get_model:
    def __init__(self) -> None:
        def get_mistral_7b(base = False):

            if base == True:
                base_model = AutoModelForCausalLM("mistralai/Mistral-7B-v0.1")
            else:
                base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
                tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            
            return base_model, tokenizer

        def get_mixtral(base=False):
            model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return base_model, tokenizer
        def get_gpt2_XL():
            model_id = "gp2-xl"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return base_model, tokenizer
        def get_gpt2_large():
            model_id = "gpt2-large"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return base_model,tokenizer
        def get_gpt2_medium():
            model_id = "gpt2-medium"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return base_model, tokenizer
        def get_gpt2():
            model_id = "gpt2"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return base_model, tokenizer
        def get_gpt():
            model_id = "openai-gpt"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return base_model, tokenizer
        def get_llama_7b(base_model=None, auth_token=""):
            model_id = ""
            base_model = LlamaForCausalLM.from_pretrained(model_id)
            tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
        def get_llama2_7b(base=False,auth_token=""):
            if base == True:
                model_id = "meta-llama/Llama-2-7b-hf"
            else:
                model_id = "meta-llama/Llama-2-7b-chat-hf"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        def get_llama2_13b(base=False,auth_token=""):
            if base == True:
                model_id = "meta-llama/Llama-2-13b-hf"
            else:
                model_id = "meta-llama/Llama-2-13b-chat-hf"
            base_model = LlamaForCausalLM.from_pretrained(model_id)
            tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
        def get_llama2_70b(base=False,auth_token=""):
            if base == True:
                model_id = "meta-llama/Llama-2-70b-hf"
            else:
                model_id = "meta-llama/Llama-2-70b-chat-hf"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        def get_bert_tiny():
            model_id = "prajjwal1/bert-tiny"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            pass
        def get_distilbert():
            model_id = "distilbert-base-uncased"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            pass
        def get_roberta():
            model_id = "roberta-uncased"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            pass
        def get_bert_base():
            model_id = "bert-base-uncased"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            pass
        def get_bert_small():
            model_id = "prajjwal1/bert-small"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            pass
        def get_bert_large():
            pass
        def get_bert_medium():
            model_id = "prajjwal1/bert-medium"
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            pass
        


        
            
        def apply_quantization():
            pass
        def use_flash_attn():
            pass
        def is_this_a_function():
            pass

