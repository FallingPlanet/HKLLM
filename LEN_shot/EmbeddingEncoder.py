from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset
from .Formatter import llama_3_formatting_func  # Ensure correct import path
import re

class EmbeddingEncoder:
    def __init__(self, model, tokenizer_name, removal_patterns=None, removal_strings=None, use_llama3_format=False):
        """
        Args:
        model (str or torch.nn.Module): Model identifier or model instance.
        tokenizer_name (str): Tokenizer identifier.
        removal_patterns (list of str): Regex patterns to remove from input texts.
        removal_strings (list of str): Exact strings to remove from input texts.
        use_llama3_format (bool): Whether to use specific formatting.
        """
        self.model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=os.getenv('HF_TOKEN'), output_hidden_states=True) if isinstance(model, str) else model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model.eval()
        self.formatting_func = None  # Assign actual formatting function if use_llama3_format is True
        self.removal_patterns = [re.compile(pattern) for pattern in removal_patterns] if removal_patterns else []
        self.removal_strings = removal_strings if removal_strings else []

    def clean_text(self, text_data):
        """Applies both string replacements and regex removals to clean text data."""
        cleaned_data = []
        for text in text_data:
            # Apply string replacements
            for removal_string in self.removal_strings:
                text = text.replace(removal_string, '')
            # Apply regex removals
            for pattern in self.removal_patterns:
                text = pattern.sub('', text)
            cleaned_data.append(text.strip())
        return cleaned_data

    def encode(self, text_data, output_path=None, return_format='numpy', batch_size=16):
        if isinstance(text_data, str):
            text_data = [text_data]

        if self.formatting_func:
            text_data = self.formatting_func(text_data, convert_from_instruction=False, convert_from_prompt_only=True, bos_token="", eos_token="")

    
        text_data = self.clean_text(text_data)

        inputs = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)
        dataloader = DataLoader(TensorDataset(inputs['input_ids'], inputs['attention_mask']), batch_size=batch_size)

        all_embeddings = []
        for batch in dataloader:
            input_ids, attention_mask = batch
            with torch.no_grad():
                model_inputs = {'input_ids': input_ids.to(self.model.device), 'attention_mask': attention_mask.to(self.model.device)}
                outputs = self.model(**model_inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1][:, 0, :]
                all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        if return_format == 'json' and output_path:
            formatted_data = [{'text': text, 'embedding': emb.tolist()} for text, emb in zip(text_data, all_embeddings)]
            with open(output_path, 'w') as f:
                json.dump(formatted_data, f)
            return formatted_data
        elif return_format == 'list':
            return all_embeddings.tolist()
        else:
            if output_path:
                np.save(output_path, all_embeddings)
        return all_embeddings


    @staticmethod
    def adapt_rlhf_data(rlhf_data):
        """ Converts RLHF data format to a simple 'prompt' and 'completion' format. """
        if isinstance(rlhf_data, dict):
            return {"prompt": rlhf_data['prompt'], "completion": rlhf_data['chosen']}
        elif isinstance(rlhf_data, list):
            return [{"prompt": item['prompt'], "completion": item['chosen']} for item in rlhf_data]
        else:
            raise TypeError("Expected rlhf_data to be a dict or list of dicts")



