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
    
def multc_multl_parser(target, text, output_type="bool"):
    if not isinstance(target, list):
        raise TypeError("target must be a list of lists.")

    if not isinstance(text, str):
        raise TypeError("The model output must be a String.")

    l_array = []
    for lst in target:
        if not isinstance(lst, list):
            raise TypeError("Each item in target must be a list.")

        c_array = []
        for element in lst:
            if not isinstance(element, str):
                raise TypeError("Each element in the sublists of target must be a String.")

            if output_type == "bool":
                c_array.append(element in text)
            elif output_type == "int":
                c_array.append(int(element in text))
            else:
                raise ValueError("output_type must be either 'bool' or 'int'.")

        l_array.append(c_array)

    return l_array

            
            
    
    
    