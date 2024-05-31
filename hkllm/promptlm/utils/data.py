import pandas as pd
import json



def prepare_dataset_for_inference(df, text_col, class_col, sample_size, max_chars=None, supp_columns=None, leading_columns=None):
    """
    Prepare a dataset for inference by sampling and optionally concatenating leading and supplementary text columns,
    handling empty or NaN values gracefully by skipping them in concatenation. Text can be truncated to a maximum length.

    Parameters:
    - df: DataFrame containing the dataset.
    - text_col: Name of the main text column.
    - class_col: Name of the classification column.
    - sample_size: Number of samples to draw from df.
    - max_chars: Optional. Maximum number of characters for each text entry.
    - supp_columns: Optional. Additional columns to concatenate after the main text.
    - leading_columns: Optional. Columns to concatenate before the main text.

    Returns:
    - A dictionary with keys 'x' for text data and 'y' for class labels.
    """
    sampled_df = df.sample(n=sample_size, replace=False)

    # Initialize combined text with leading columns if present
    if leading_columns:
        sampled_df['combined_text'] = sampled_df[leading_columns].astype(str).agg(lambda x: ' '.join(x.dropna()), axis=1)
        sampled_df['combined_text'] += " " + sampled_df[text_col].astype(str)
    else:
        sampled_df['combined_text'] = sampled_df[text_col].astype(str)

    # Append supplementary columns, skipping blanks and NaNs
    if supp_columns:
        for col in supp_columns:
            sampled_df['combined_text'] += " " + sampled_df[col].astype(str).replace(r'^\s*$', '', regex=True).apply(lambda x: x if x != '' else '')

    # Truncate text to the specified maximum number of characters
    if max_chars is not None:
        sampled_df['combined_text'] = sampled_df['combined_text'].apply(lambda x: x[:max_chars])

    x_data = sampled_df['combined_text'].tolist()
    data_for_inference = {
        'x': x_data,
        'y': sampled_df[class_col].tolist()
    }
    return data_for_inference




def prepare_dataset_for_conversion(df, text_col, class_col,annotation_col, sample_size, supp_columns=None, leading_columns=None):
    if annotation_col == None:
        print("No annotation column has been passed, if this is not in error, please use <prepare_dataset_for_inference>")
    """
    Prepare a dataset for inference by sampling and optionally concatenating leading and supplementary text columns,
    handling empty or NaN values gracefully by skipping them in concatenation.

    Parameters:
    - df: DataFrame containing the dataset.
    - text_col: Name of the main text column.
    - class_col: Name of the classification column.
    - sample_size: Number of samples to draw from df.
    - supp_columns: Optional. Additional columns to concatenate after the main text.
    - leading_columns: Optional. Columns to concatenate before the main text.

    Returns:
    - A dictionary with keys 'x' for text data , 'y' for class labels and 'z' for the annotations.
    
    
    Notes:
    - This will not work with the normal parser utilize the conversion_parser from hkllm.promptlm.utils.parsers
    """
    
    
    sampled_df = df.sample(n=sample_size, replace=False)

    # Initialize combined text with leading columns if present
    if leading_columns:
        sampled_df['combined_text'] = sampled_df[leading_columns].astype(str).agg(lambda x: ' '.join(x.dropna()), axis=1)
        sampled_df['combined_text'] += " " + sampled_df[text_col].astype(str)
    else:
        sampled_df['combined_text'] = sampled_df[text_col].astype(str)

    # Append supplementary columns, skipping blanks and NaNs
    if supp_columns:
        for col in supp_columns:
            sampled_df['combined_text'] += sampled_df[col].astype(str).replace(r'^\s*$', '', regex=True).apply(lambda x: ' ' + x if x != '' else '')

    x_data = sampled_df['combined_text'].tolist()

    data_for_conversion = {
        'x': x_data,
        'y': sampled_df[class_col].tolist(),
        'z': sampled_df[annotation_col]
        
    }
    return data_for_conversion

def prepare_dataset_for_generator(df, indices_csv_path, indices_column_name, text_col, class_col=None, sample_size=100, supp_columns=None, leading_columns=None):
    """
    Prepare a dataset for generator tasks by excluding specific indices, sampling, and optionally concatenating text columns,
    handling empty or NaN values gracefully by skipping them in concatenation.

    Parameters:
    - df: DataFrame containing the dataset.
    - indices_csv_path: Path to a CSV file containing indices to exclude from sampling.
    - indices_column_name: The name of the column in the CSV that contains the indices to exclude.
    - text_col: Name of the column in df that contains text data.
    - class_col: Optional. Name of the column in df that classifies the text data.
    - sample_size: Number of samples to draw from df.
    - supp_columns: Additional columns to concatenate to the text after the main text.
    - leading_columns: Columns to concatenate to the text before the main text.

    Returns:
    - A dictionary with keys 'x' for text data, 'y' for class labels (if class_col is not None), and 'Index' for the indices of the sampled rows.
    """
    # Load indices to exclude from sampling
    exclude_df = pd.read_csv(indices_csv_path)
    exclude_indices = exclude_df[indices_column_name].tolist()

    # Exclude specified indices from the DataFrame
    df_filtered = df.drop(exclude_indices, errors='ignore')

    # Sample the DataFrame
    sampled_df = df_filtered.sample(n=sample_size, replace=False)

    # Initialize combined text with leading columns if present
    if leading_columns:
        sampled_df['combined_text'] = sampled_df[leading_columns].astype(str).agg(lambda x: ' '.join(x.dropna()), axis=1)
        sampled_df['combined_text'] += " " + sampled_df[text_col].astype(str)
    else:
        sampled_df['combined_text'] = sampled_df[text_col].astype(str)

    # Append supplementary columns, skipping blanks and NaNs
    if supp_columns:
        for col in supp_columns:
            sampled_df['combined_text'] += sampled_df[col].astype(str).replace(r'^\s*$', '', regex=True).apply(lambda x: ' ' + x if x != '' else '')

    x_data = sampled_df['combined_text'].tolist()

    # Prepare the data for generator tasks
    data_for_generator = {'x': x_data, 'Index': sampled_df.index.tolist()}
    if class_col:
        data_for_generator['y'] = sampled_df[class_col].tolist()

    return data_for_generator


def extract_generated_text(full_text, prompt):
    """
    Extracts the generated text from the full text by removing the prompt.

    Parameters:
    - full_text: The complete text returned by the model, which includes the prompt and the generated text.
    - prompt: The prompt that was originally provided to the model.

    Returns:
    - The generated text, with the prompt removed.
    """
    if full_text.startswith(prompt):
        # Remove the prompt from the full text to isolate the generated text
        return full_text[len(prompt):]
    else:
        # If the full text does not start with the prompt, return the full text as is
        return full_text

def rlhf_sample(prompt, chosen, rejected):
    """
    Generates a dictionary for a RLHF sample including the prompt, one accepted completion, and one rejected completion.

    Args:
    - prompt (str): The input prompt for the sample.
    - accepted (str): The accepted completion for the prompt.
    - rejected (str): The rejected completion for the prompt.

    Returns:
    - dict: A dictionary representing the RLHF sample.
    """
    sample = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }
    return sample

def create_instruction_pair(prompt, completion):
    """
    Generates a dictionary for an instruction pair including the prompt and its completion.

    Args:
    - prompt (str): The input prompt.
    - completion (str): The expected completion for the prompt.

    Returns:
    - dict: A dictionary representing the instruction pair.
    """
    instruction_pair = {
        "prompt": prompt,
        "completion": completion
    }
    return instruction_pair

def update_indices_csv(csv_path, indices_column_name, new_index):
    """
    Update the CSV file with a new index to exclude from future sampling, one at a time.

    Parameters:
    - csv_path: Path to the CSV file containing indices to exclude.
    - indices_column_name: The name of the column in the CSV that contains the indices to exclude.
    - new_index: A single new index to append to the CSV.
    """
    # Load the existing CSV file or initialize an empty DataFrame if the file does not exist
    try:
        existing_indices_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        existing_indices_df = pd.DataFrame(columns=[indices_column_name])
    
    # Append the new index
    new_index_df = pd.DataFrame([new_index], columns=[indices_column_name])
    updated_indices_df = pd.concat([existing_indices_df, new_index_df]).drop_duplicates().reset_index(drop=True)
    
    # Save the updated DataFrame back to CSV
    updated_indices_df.to_csv(csv_path, index=False)

    
def generate_shot_examples(data_dict,   shot_examples):
    
    df = pd.DataFrame(data_dict)
    
    shot_examples_dict = {'x': [], 'y':[]}
    
    indicies_to_remove = []
    
    unique_classes = df['y'].unique()
    
    for cls in unique_classes:
        class_samples = df[df['y'] == cls]
        
        n_samples = min(len(class_samples),shot_examples)
        sampled = class_samples.sample(n_samples)
        
        shot_examples_dict['x'].extend(sampled['x'].tolist())
        shot_examples_dict['y'].extend(sampled['y'].tolist())
        
        indicies_to_remove.extend(sampled.index.tolist())
    
    df_remaining = df.drop(indicies_to_remove)
    
    remaining_data_dict = df_remaining.to_dict('list')
    
    return shot_examples_dict,remaining_data_dict
        
        
    
    
import json
import os

def append_to_json_file(file_path, new_data):
    """
    Appends a new dictionary to a JSON file that contains a list of dictionaries.
    
    Args:
    - file_path (str): Path to the JSON file.
    - new_data (dict): New data to append.
    """
    # Check if the file exists and has content
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r+') as file:
            # Load the existing data
            file_data = json.load(file)
            # Append the new data
            file_data.append(new_data)
            # Reset the file pointer to the beginning of the file
            file.seek(0)
            # Write the updated data back to the file
            json.dump(file_data, file, indent=4)
            # Truncate the file to the new size in case the new data is smaller than the old
            file.truncate()
    else:
        # If the file doesn't exist or is empty, create it with the new_data as the first entry
        with open(file_path, 'w') as file:
            json.dump([new_data], file, indent=4)
            
            
def json_to_generative_dataset():
    pass
    
    

def prepare_dataset_for_training():
    pass


def strip_output_column_by_index(df, input_col_index, output_col_index):
    def strip_repeated_input(input_text, output_text):
        # Find the start of the repeated prompt in the completion
        start_index = output_text.find(input_text)
        # If the prompt is found in the completion, remove it
        if start_index != -1:
            # Remove the prompt and return the completion starting after the prompt
            return output_text[:start_index] + output_text[start_index + len(input_text):]
        return output_text
    
    # Use iloc to access the DataFrame by column indices
    for i in range(len(df)):
        input_text = df.iloc[i, input_col_index]
        output_text = df.iloc[i, output_col_index]
        df.iloc[i, output_col_index] = strip_repeated_input(input_text, output_text)
    return df

import os
def append_to_json_file(file_path, new_data):
    """
    Appends a new dictionary to a JSON file that contains a list of dictionaries.
    
    Args:
    - file_path (str): Path to the JSON file.
    - new_data (dict): New data to append.
    """
    # Check if the file exists and has content
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r+') as file:
            # Load the existing data
            file_data = json.load(file)
            # Append the new data
            file_data.append(new_data)
            # Reset the file pointer to the beginning of the file
            file.seek(0)
            # Write the updated data back to the file
            json.dump(file_data, file, indent=4)
            # Truncate the file to the new size in case the new data is smaller than the old
            file.truncate()
    else:
        # If the file doesn't exist or is empty, create it with the new_data as the first entry
        with open(file_path, 'w') as file:
            json.dump([new_data], file, indent=4)

def change_keys(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for i in range(len(data)):
        data[i]['chosen'] = data[i]['accepted']
        del data[i]['accepted']

    with open(json_file, 'w') as f:
        json.dump(data, f)


