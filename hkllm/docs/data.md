###Documentation for 'prepare_dataset_for_inference'
## `prepare_dataset_for_inference`

Prepares a dataset for inference by sampling and optionally concatenating additional text columns, handling empty or NaN values gracefully.

### Parameters
- **df** (`pandas.DataFrame`): The DataFrame containing the dataset.
- **text_col** (`str`): Name of the main text column.
- **class_col** (`str`): Name of the classification column.
- **sample_size** (`int`): Number of samples to draw from df.
- **supp_columns** (`list` of `str`, optional): Additional columns to concatenate after the main text.
- **leading_columns** (`list` of `str`, optional): Columns to concatenate before the main text.

### Returns
- **dict**: A dictionary with keys 'x' for text data and 'y' for class labels.

### Example
```python
df = pd.DataFrame({
    'title': ["Title 1", "Title 2"],
    'description': ["Desc 1", "Desc 2"],
    'category': ["Category 1", "Category 2"]
})
prepared_data = prepare_dataset_for_inference(df, 'description', 'category', 2)
print(prepared_data)
```
### Documentation for `prepare_dataset_for_conversion`


## `prepare_dataset_for_conversion`

Prepares a dataset for conversion tasks by sampling and concatenating text columns, similar to `prepare_dataset_for_inference`, but includes an annotation column.

### Parameters
- **df** (`pandas.DataFrame`): The DataFrame containing the dataset.
- **text_col** (`str`): Name of the main text column.
- **class_col** (`str`): Name of the classification column.
- **annotation_col** (`str`): Name of the column containing annotations.
- **sample_size** (`int`): Number of samples to draw.
- **supp_columns** (`list` of `str`, optional): Additional columns to concatenate after the main text.
- **leading_columns** (`list` of `str`, optional): Columns to concatenate before the main text.

### Returns
- **dict**: A dictionary with keys 'x' for text data, 'y' for class labels, and 'z' for annotations.

### Notes
- Utilizes `conversion_parser` from `hkllm.promptlm.utils.parsers` for parsing.

### Example

```python
# Assume df and parameters are defined as shown in previous examples
data_for_conversion = prepare_dataset_for_conversion(df, 'description', 'category', 'notes', 2)
print(data_for_conversion)
```
### Documentation for `prepare_dataset_for_generator`


## `prepare_dataset_for_generator`

Prepares a dataset for generator tasks by excluding specific indices and sampling, optionally concatenating text columns.

### Parameters
- **df** (`pandas.DataFrame`): The DataFrame containing the dataset.
- **indices_csv_path** (`str`): Path to a CSV file containing indices to exclude.
- **indices_column_name** (`str`): Column name in the CSV with indices.
- **text_col** (`str`): Main text column name.
- **class_col** (`str`, optional): Classification column name.
- **sample_size** (`int`): Number of samples to draw.
- **supp_columns** (`list` of `str`, optional): Additional columns to concatenate after the main text.
- **leading_columns** (`list` of `str`, optional): Columns to concatenate before the main text.

### Returns
- **dict**: A dictionary with keys 'x' for text data, 'y' for class labels (if class_col is not None), and 'Index' for sampled row indices.

### Example
```python
# Example usage assuming the DataFrame and CSV path are correctly set up
data_for_generator = prepare_dataset_for_generator(df, 'path/to/indices.csv', 'exclude_indices', 'description', 'category', 100)
print(data_for_generator)
```
## `extract_generated_text`

Extracts the generated text from a complete text string by removing the specified prompt.

### Parameters
- **full_text** (`str`): The complete text returned by the model, including the prompt and the generated text.
- **prompt** (`str`): The prompt that was originally provided to the model.

### Returns
- **str**: The generated text after the prompt has been removed. Returns the full text as is if the prompt does not match the beginning of the full text.

### Example
```python
full_text = "Hello world, how are you? Hello world,"
prompt = "Hello world,"
generated_text = extract_generated_text(full_text, prompt)
# Output: " how are you? Hello world,"
```
### Documentation for `rlhf_sample`


## `rlhf_sample`

Generates a dictionary representing a Reinforcement Learning from Human Feedback (RLHF) sample, including a prompt and choices between an accepted and a rejected completion.

### Parameters
- **prompt** (`str`): The input prompt for the RLHF task.
- **chosen** (`str`): The completion that has been chosen or accepted.
- **rejected** (`str`): The completion that has been rejected.

### Returns
- **dict**: A dictionary with keys 'prompt', 'chosen', and 'rejected' representing the RLHF sample.

### Example
```python
rlhf_dict = rlhf_sample("What is the capital of France?", "Paris", "Lyon")
# Output: {'prompt': 'What is the capital of France?', 'chosen': 'Paris', 'rejected': 'Lyon'}
```
### Documentation for `create_instruction_pair`


## `create_instruction_pair`

Creates a dictionary representing an instructional pair, which includes a prompt and its corresponding expected completion.

### Parameters
- **prompt** (`str`): The input prompt.
- **completion** (`str`): The expected completion for the prompt.

### Returns
- **dict**: A dictionary with keys 'prompt' and 'completion' representing the instructional pair.

### Example
```python
instruction = create_instruction_pair("Translate to French: Hello", "Bonjour")
# Output: {'prompt': 'Translate to French: Hello', 'completion': 'Bonjour'}
```
### Documentation for `update_indices_csv`


## `update_indices_csv`

Updates a CSV file by appending a new index to a specified column. This is typically used to exclude certain indices from future processing or sampling tasks.

### Parameters
- **csv_path** (`str`): Path to the CSV file containing indices to exclude.
- **indices_column_name** (`str`): The name of the column in the CSV that contains the indices to exclude.
- **new_index** (`int`): The new index to append to the CSV file.

### Returns
- None: Updates the CSV file directly.

### Example
```python
update_indices_csv("path/to/indices.csv", "exclude_indices", 102)
# The file "path/to/indices.csv" now includes the new index 102 in the "exclude_indices" column.
```
### Documentation for `generate_shot_examples`


## `generate_shot_examples`

Generates few-shot learning examples from a given dataset dictionary, sampling a specified number of examples for each unique class.

### Parameters
- **data_dict** (`dict`): A dictionary containing data with keys 'x' for text data and 'y' for labels.
- **shot_examples** (`int`): Number of examples to sample for each class.

### Returns
- **tuple**:
  - First element (`dict`): A dictionary with few-shot examples.
  - Second element (`dict`): The remaining data after few-shot examples have been removed.

### Example
```python
data_dict = {'x': ['text1', 'text2', 'text3'], 'y': ['class1', 'class2', 'class1']}
shot_examples, remaining_data = generate_shot_examples(data_dict, 1)
# Outputs shot_examples with 1 example per class and the remaining data
