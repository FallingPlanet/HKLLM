
def sample_accuracy(y_true, y_pred, all_or_nothing=False):
    if len(y_true) != len(y_pred):
        raise ValueError("Predictions and Ground Truth must be Lists of the same list")
    count =  0
    accuracy = 0
    for x, y in zip(y_true,y_pred):
        if x != y:
            count +=1
        
    if all_or_nothing == False:
        accuracy = 1 - (count/len(y_pred))
    else: 
        if count == 1: accuracy = 0
    return accuracy
            
    