class ChatTemplateCreator:
    def __init__(self, embedding_encoder, calculate, tokenizer, combined_data, num_examples=3, example_role='assistant', llama_3=True, **kwargs):
        """
        Initializes the ChatTemplateCreator with essential components for generating chat templates.
        
        Args:
            embedding_encoder (EmbeddingEncoder): An instance for generating embeddings.
            calculate (Calculate): An instance for computing distances between embeddings.
            tokenizer: Tokenizer for processing text inputs.
            combined_data (list of dicts): Contains both embeddings and associated textual data.
            num_examples (int): Number of n-shot examples to include in the prompt.
            example_role (str): Specifies whether the examples should be presented as spoken by the 'user', 'assistant', or alternating between them.
            llama_3 (bool): Specifies if the Llama-3 formatting should be applied.
            **kwargs: Additional keyword arguments, including 'distance_type' for specifying the type of distance metric to use (default is 'cosine').
        """
        self.embedding_encoder = embedding_encoder
        self.calculate = calculate
        self.tokenizer = tokenizer
        self.combined_data = combined_data
        self.num_examples = num_examples
        self.example_role = example_role
        self.llama_3 = llama_3
        self.distance_type = kwargs.get('distance_type', 'cosine')

    def create_template(self, text, system_prompt, user_prompt, follow_up_prompt, end_of_example_text="End of examples."):
        """
        Creates a chat template by selecting relevant examples based on the input text and integrating them into a conversational format.
        
        Args:
            text (str): Target text for which to create the template.
            system_prompt (str): System prompt to be included at the beginning of the template.
            user_prompt (str): User prompt to introduce the user's input.
            follow_up_prompt (str): Follow-up prompt to append after the user's input.
            end_of_example_text (str): Text to indicate the end of example section.

        Returns:
            str: A fully formatted chat template incorporating system prompts, user prompts, and n-shot examples.
        """
        target_embedding = self.embedding_encoder.encode([text])[0]
        distances = [(self.calculate.compute(target_embedding, entry['embedding'], self.distance_type), idx) for idx, entry in enumerate(self.combined_data)]
        selected_indices = [idx for _, idx in sorted(distances)[:self.num_examples]]
        examples_text = self.format_examples(selected_indices)
        return self.assemble_chat_template(examples_text, text, system_prompt, user_prompt, follow_up_prompt, end_of_example_text)

    def format_examples(self, selected_indices):
        """
        Formats the selected examples to be included in the chat template.
        
        Args:
            selected_indices (list of int): Indices of the entries selected as examples.

        Returns:
            list of dict: Formatted examples ready to be inserted into the chat template.
        """
        formatted_examples = []
        for idx in selected_indices:
            entry = self.combined_data[idx]
            role = "assistant" if self.example_role == 'assistant' else "user"
            formatted_examples.append({"role": role, "content": f"{entry['prompt']}\n{entry['completion']}"})
        return formatted_examples

    def assemble_chat_template(self, examples_text, user_input, system_prompt, user_prompt, follow_up_prompt, end_of_example_text):
        """
        Assembles the chat template from various components including system prompts, user input, and example interactions.
        
        Args:
            examples_text (list of dict): Prepared examples to be included in the chat template.
            user_input (str): User's input to be included in the template.
            system_prompt (str): System prompt at the beginning of the template.
            user_prompt (str): User prompt that leads into the user's input.
            follow_up_prompt (str): Follow-up prompt after the user's input.
            end_of_example_text (str): Text indicating the end of the examples section.

        Returns:
            str: The complete chat template ready for use.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "system", "content": "### Examples:"}
        ] + examples_text
        messages.append({"role": "user", "content": end_of_example_text})
        messages += [
            {"role": "user", "content": user_input},
            {"role": "user", "content": follow_up_prompt}
        ]
        if self.llama_3:
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            examples_section = "\n\n### Examples:\n" + "\n".join([ex['content'] for ex in examples_text]) + "\n\n" + end_of_example_text
            full_text = f"{system_prompt}\n{user_prompt}\n{examples_section}\n{user_input}\n{follow_up_prompt}"
        return full_text