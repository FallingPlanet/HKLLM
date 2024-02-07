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
    unique_classes = df['y'].unique()
    
    shot_examples_dict = {'x': [], 'y':[]
    }
    for cls in unique_classes:
        class_samples = df[df['y']] == cls
        
        sampled = class_samples.sample(min(len(class_samples),shot_examples))
        
        shot_examples_dict['x'].extend(sampled['x'].tolist())
        shot_examples_dict['y'].extend(sampled['y'].tolist())
        
        return shot_examples_dict
    pass

def prepare_dataset_for_training():
    pass