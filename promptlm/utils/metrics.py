
def sample_accuracy(y_true, y_pred, all_or_nothing=False,mc_ml=False):
    if mc_ml == True:
        if len(y_true) != len(y_pred):
            raise ValueError("Predictions and Ground Truth must be Lists of the same length")
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
    else:
        if len(y_true) != len(y_pred):
            raise ValueError("Preidctions and Ground Truth must be lists of the same length")
        
                
    