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


import re

def parse_output_for_answer(output, keywords, single_output=True):
    # Allowing for optional spaces within the tag brackets
    keyword_patterns = [fr"<\s*{keyword}\s*>" for keyword in keywords]
    patterns = [
        r'<\s*Tag\s*>\s*\[(.*?)\]\s*<\/\s*Tag\s*>',
        r'<\s*tag\s*>\s*\[(.*?)\]\s*<\/\s*tag\s*>',
        r'<\s*Tags\s*>\s*\[(.*?)\]\s*<\/\s*Tags\s*>',
        r'<\s*Tag\s*>\s*\[(.*?)\]\s*<\/\s*Tags\s*>',
        r'<\s*TAG\s*>\s*\[(.*?)\]\s*<\/\s*TAG\s*>',
        r'<\s*TAGS\s*>\s*\[(.*?)\]\s*<\/\s*TAGS\s*>',
        r'<\s*TAG\s*>\s*\[(.*?)\]\s*<\/\s*TAGS\s*>',
        r'<\s*TAG\s*>\s*\[(.*?)\]\s*<\/\s*Tag\s*>',
        r'<\s*Tag\s*>\s*(.*?)\s*<\/\s*Tag\s*>',
        r'<\s*tag\s*>\s*(.*?)\s*<\/\s*tag\s*>',
        r'<\s*Tag\s*>\s*(.*?)\s*<\/\s*Tags\s*>',
        r'<\s*TAG\s*>\s*(.*?)\s*<\/\s*TAG\s*>',
        r'<\s*TAG\s*>\s*(.*?)\s*<\/\s*Tag\s*>',
        r'<\s*TAG\s*>\s*(.*?)\s*<\/\s*TAGs\s*>',
        r'<\s*Answer\s*>\s*(.*?)\s*<\/\s*Answer\s*>',
        r'\[\s*(.*?)\s*\]',
        # Patterns below are adjusted to allow optional spaces within tags
        r'<\s*(.*?)\s*>',
        r'<([^<>]+)>',
        r'<([^>]+)>',
        r'(?:Your response:|Answer:)\s*(.*?)(?=:[\.,\,]|$)'
    ]
    pattern = '|'.join(patterns)
    
    matches = re.findall(pattern, output, re.DOTALL)
    print("found matches = ", matches)
    
    extracted_answers = []
    for match in matches:
        filtered_match = list(filter(None, match))
        if filtered_match:
            answer = filtered_match[0].strip()
            if not keywords or any(keyword.lower() in answer.lower() for keyword in keywords):
                extracted_answers.append(answer)
    return extracted_answers

    

            
            
    
    
    