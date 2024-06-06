from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset

class EmbeddingEncoder:
    def __init__(self, model, tokenizer_name=None, device=None, adapter_name=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        token = os.getenv('HF_TOKEN')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=token, device_map="auto", output_hidden_states=True)
            if adapter_name:
                adapter_source = 'hf' if os.path.exists(adapter_name) else 'local'
                self.model.load_adapter(adapter_name, source=adapter_source)
        else:
            self.model = model.to(self.device)

        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.model.eval()

    def encode(self, text_data, output_path=None, return_format='numpy', text_column=None, batch_size=16):
        if isinstance(text_data, dict) and text_column is None:
            raise ValueError("text_column must be specified if text_data is a dictionary")
        
        text_data = [text_data] if isinstance(text_data, str) else text_data.get(text_column, text_data) if isinstance(text_data, dict) else text_data

        inputs = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        dataset = TensorDataset(inputs['input_ids'], inputs.get('attention_mask', torch.Tensor()))
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_embeddings = []

        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.device)
            if attention_mask.nelement() != 0:
                attention_mask = attention_mask.to(self.device)
                model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            else:
                model_inputs = {'input_ids': input_ids}

            with torch.no_grad():
                outputs = self.model(**model_inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                embeddings = hidden_states[-1][:, 0, :]
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        if return_format == 'json':
            formatted_data = [{'text': text, 'embedding': emb.tolist()} for text, emb in zip(text_data, all_embeddings)]
            if output_path:
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
        """
        Adapts RLHF data to a format suitable for other components, converting 'chosen' to 'completion'.
        It operates non-destructively, returning a new dictionary with the desired structure.
        
        Args:
            rlhf_data (dict or list of dicts): RLHF data in the format with 'prompt', 'chosen', and 'rejected'.

        Returns:
            dict or list of dicts: New data formatted with 'prompt' and 'completion' keys, original data remains unchanged.
        """
        if isinstance(rlhf_data, dict):
            # Handle single dictionary case, create a new dictionary
            return {
                "prompt": rlhf_data['prompt'],
                "completion": rlhf_data['chosen']
            }
        elif isinstance(rlhf_data, list):
            # Handle list of dictionaries case, create new dictionaries for each item
            return [{
                "prompt": item['prompt'],
                "completion": item['chosen']
            } for item in rlhf_data]
        else:
            raise TypeError("Expected rlhf_data to be a dict or list of dicts")

