import numpy as np
from .Calculate import Calculate
from .EmbeddingEncoder import EmbeddingEncoder
import json

import numpy as np

class ChatTemplateCreator:
    def __init__(self, embedding_encoder, calculate, embedding_path, num_examples=3, example_role='assistant', llama_3=True):
        """
        Initializes the ChatTemplateCreator with an EmbeddingEncoder and a Calculate instance for embedding and distance calculations.
        
        Args:
            embedding_encoder (EmbeddingEncoder): An instance of EmbeddingEncoder for generating embeddings.
            calculate (Calculate): An instance of Calculate for computing distances.
            embedding_path (str): Path to the .npy file containing precomputed embeddings.
            num_examples (int): Number of n-shot examples to include in the prompt.
            example_role (str): Role for the n-shot examples ('user', 'assistant', or 'alternate').
            llama_3 (bool): Whether to use Llama-3 chat template formatting.
        """
        self.embedding_encoder = embedding_encoder
        self.calculate = calculate
        self.embeddings = np.load(embedding_path)
        self.num_examples = num_examples
        self.example_role = example_role
        self.llama_3 = llama_3

    def create_template(self, text, system_prompt, user_prompt, follow_up_prompt):
        """
        Creates a formatted chat template using a target text to find the most similar examples based on embedding distances.
        
        Args:
            text (str): The target text to encode and compare.
            system_prompt (str): System prompt introducing the task.
            user_prompt (str): User prompt before displaying the examples.
            follow_up_prompt (str): Follow-up prompt to complete the interaction.

        Returns:
            str: A formatted chat template.
        """
        target_embedding = self.embedding_encoder.encode([text])[0]  # Generate embedding for the new sample text
        distances = [(self.calculate.compute(target_embedding, emb), idx) for idx, emb in enumerate(self.embeddings)]
        selected_indices = [idx for _, idx in sorted(distances)[:self.num_examples]]  # Select the indices of the most similar embeddings
        examples_text = self.format_examples(selected_indices)

        return self.assemble_chat_template(examples_text, text, system_prompt, user_prompt, follow_up_prompt)

    def format_examples(self, selected_indices):
        """
        Formats selected examples based on indices.
        """
        formatted_examples = []
        for idx in selected_indices:
            prompt = self.embeddings[idx]['prompt']
            completion = self.embeddings[idx]['completion']
            if self.example_role == 'alternate':
                formatted_examples.append({"role": "user", "content": prompt})
                formatted_examples.append({"role": "assistant", "content": completion})
            else:
                role = "assistant" if self.example_role == 'assistant' else "user"
                formatted_examples.append({"role": role, "content": f"{prompt}\n{completion}"})
        return formatted_examples

    def assemble_chat_template(self, examples_text, user_input, system_prompt, user_prompt, follow_up_prompt):
        """
        Assembles the complete chat template including system prompts and user interactions.
        
        Args:
            examples_text (list): Formatted text examples.
            user_input (str): The actual text input by the user.
            system_prompt (str): System prompt to introduce the context.
            user_prompt (str): Initial user prompt before showing examples.
            follow_up_prompt (str): Follow-up to ask for user interaction.
        
        Returns:
            str: The fully assembled chat template.
        """
        if self.llama_3:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "user", "content": user_input}  # Insert the user's text here
            ] + examples_text
            messages.append({"role": "user", "content": follow_up_prompt})
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            examples_section = "\n".join([ex['content'] for ex in examples_text])
            full_text = f"{system_prompt}\n{user_prompt}\n{user_input}\n{examples_section}\n{follow_up_prompt}"
        return full_text

