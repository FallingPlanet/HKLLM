import pandas as pd

def prepare_dataset_for_inference(df, text_col, class_col, sample_size):
    sampled_df = df.sample(n=sample_size, replace=False)
    
    data_for_inference = {
        'x': sampled_df[text_col].tolist(),
        'y': sampled_df[class_col].tolist()
    }
    return data_for_inference
    

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
        
        
    
    
    

def prepare_dataset_for_training():
    pass