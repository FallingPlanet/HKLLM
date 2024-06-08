from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset
from .Formatter import llama_3_formatting_func  # Ensure correct import path
import re

class EmbeddingEncoder:
    def __init__(self, model, tokenizer_name=None, adapter_name=None, use_llama3_format=False, removal_texts=None):
        """Initializes the EmbeddingEncoder with specific configurations for the model, tokenizer, and formatting function.

        Args:
        model (str or torch.nn.Module): The pre-trained model or the path to the pre-trained model directory.
        tokenizer_name (str, optional): The tokenizer name or path used for tokenizing input texts.
        adapter_name (str, optional): The path or identifier for a pre-trained adapter to be loaded into the model.
        use_llama3_format (bool, optional): Flag to determine whether to preprocess text data using the Llama-3 formatting function before encoding.
        removal_texts (list of str, optional): List of texts to remove from input data.
        """
        self.model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=os.getenv('HF_TOKEN'), device_map="auto", output_hidden_states=True) if isinstance(model, str) else model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model.eval()
        self.formatting_func = llama_3_formatting_func if use_llama3_format else None
        self.removal_texts = [re.compile(re.escape(text)) for text in removal_texts] if removal_texts else []

    def clean_text(self, text_data):
        """ Cleans prompt information from the text data by removing specified blocks of text. """
        cleaned_data = []
        for text in text_data:
            for pattern in self.removal_texts:
                text = pattern.sub('', text)  # Apply each regex pattern separately
            cleaned_data.append(text.strip())
        return cleaned_data

    def encode(self, text_data, output_path=None, return_format='numpy', batch_size=16):
        if isinstance(text_data, str):
            text_data = [text_data]

        if self.formatting_func:
            text_data = self.formatting_func(text_data, convert_from_instruction=False, convert_from_prompt_only=True, bos_token="", eos_token="")

        if self.removal_texts:
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



