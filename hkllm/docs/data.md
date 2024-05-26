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
