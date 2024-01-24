from transformers import AutoModelForCausalLM, AutoTokenizer

class get_7b:
    def get_model():
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        return base_model, tokenizer