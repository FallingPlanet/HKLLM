def instruction_to_messages(dialog_entries):
    messages_entries = []
    for entry in dialog_entries:
        messages = [
            {"role": "user", "content": entry["prompt"]},
            {"role": "assistant", "content": entry["completion"]}
        ]
        messages_entries.append({"messages": messages})
    return messages_entries

def prompt_only_to_messages(prompts):
    messages_entries = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        messages_entries.append({"messages": messages})
    return messages_entries

def llama_3_formatting_func(dialog_entries, convert_from_instruction=False, convert_from_prompt_only=False, bos_token="", eos_token=""):
    if convert_from_instruction and convert_from_prompt_only:
        raise ValueError("Only one conversion method can be specified.")

    if convert_from_instruction:
        dialog_entries = instruction_to_messages(dialog_entries)
    elif convert_from_prompt_only:
        dialog_entries = prompt_only_to_messages(dialog_entries)

    formatted_entries = []
    for entry in dialog_entries:
        formatted_text = bos_token
        for message in entry["messages"]:
            formatted_text += f"{message['role']}\n\n{message['content'].strip()}\n"
        formatted_text += eos_token
        formatted_entries.append(formatted_text)

    return formatted_entries
