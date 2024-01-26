def multc_parser(target, text, output_type = "bool"):
    if not isinstance(target, list):
        raise TypeError("target must be a list of classes")
    if not isinstance(text, str):
        raise TypeError("model output must be a String.")
    for x in target:
        if not isinstance(x, str):
            raise TypeError("All elements in the target must be a string")
        
    b_array = []
    if output_type == "bool":
        for element in target:
            if element in text: b_array.append(True)
            else: b_array.append(False)
        return b_array
    if output_type == "int":
        for element in target:
            if element in text: b_array.append(1)
            else: b_array.append(0)
        return b_array
    
    
    