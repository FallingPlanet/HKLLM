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
    
    # Updated patterns to better handle mismatches and broken tags
    patterns = [
        r'<\s*Tag\s*>\s*\[(.*?)\]\s*<\/\s*Tag\s*>',
        r'<\s*tag\s*>\s*\[(.*?)\]\s*<\/\s*tag\s*>',
        r'<\s*Tags\s*>\s*\[(.*?)\]\s*<\/\s*Tags\s*>',
        r'<\s*Tag\s*>\s*\[(.*?)\]\s*<\/\s*Tags\s*>',
        r'<\s*TAG\s*>\s*\[(.*?)\]\s*<\/\s*TAG\s*>',
        r'<\s*TAGS\s*>\s*\[(.*?)\]\s*<\/\s*TAGS\s*>',
        r'<\s*TAG\s*>\s*\[(.*?)\]\s*<\/\s*TAGS\s*>',
        r'<\s*TAG\s*>\s*\[(.*?)\]\s*<\/\s*Tag\s*>',
        r'<\s*Tag\s*>\s*(.*?)\s*<\/\s*Tag\s*\s*>',
        r'<\s*tag\s*>\s*(.*?)\s*<\/\s*tag\s*\s*>',
        r'<\s*Tag\s*>\s*(.*?)\s*<\/\s*Tags\s*\s*>',
        r'<\s*TAG\s*>\s*(.*?)\s*<\/\s*TAG\s*\s*>',
        r'<\s*TAG\s*>\s*(.*?)\s*<\/\s*Tag\s*\s*>',
        r'<\s*TAG\s*>\s*(.*?)\s*<\/\s*TAGs\s*\s*>',
        r'<\s*Answer\s*>\s*(.*?)\s*<\/\s*Answer\s*>',
        r'\[\s*(.*?)\s*\]',
        # Patterns to catch broken or incorrectly closed tags
        r'<\s*(.*?)\s*>\s*(.*?)\s*<\/\s*\1\s*>',
        r'<([^<>]+)>\s*(.*?)\s*<\/\s*\1\s*>',
        r'<([^>]+)>\s*(.*?)\s*<\/\s*\1\s*>',
        r'(?:Your response:|Answer:)\s*(.*?)(?=:[\.,\,]|$)'
    ]
    
    pattern = '|'.join(patterns)
    
    matches = re.findall(pattern, output, re.DOTALL)
    print("found matches = ", matches)
    
    extracted_answers = []
    for match in matches:
        # Flatten tuple matches to find non-empty strings
        filtered_match = [m for submatch in match if submatch for m in submatch.split() if m.strip()]
        if filtered_match:
            answer = filtered_match[0].strip()
            if not keywords or any(keyword.lower() in answer.lower() for keyword in keywords):
                extracted_answers.append(answer)
    
    return extracted_answers
def decompose_tag(data_dict, key, exact_tags):
    """
    Decompose strings in a list within a dictionary by extracting exact tags and the remainder of the string for each.

    Parameters:
    data_dict (dict): Dictionary containing lists of strings under various keys.
    key (str): The key in the dictionary whose list is to be processed.
    exact_tags (list of str): Strings considered as tags.

    Returns:
    dict:
        A dictionary with decomposed data (tag and remainder) for each original string under the specified key.
    """
    results = []
    strings_to_process = data_dict.get(key, [])
    
    # Compile regex to match any of the exact_tags at the start of each string entry
    tag_regex = re.compile(r'(' + '|'.join(map(re.escape, exact_tags)) + ')(.*)')
    
    for item in strings_to_process:
        individual_entries = item.strip("[]").replace("'", "").split(",")
        decomposed_entries = []
        for entry in individual_entries:
            entry = entry.strip()
            match = tag_regex.match(entry)
            if match:
                found_tag, remainder = match.groups()
                decomposed_entries.append((found_tag, remainder))
        results.append(decomposed_entries)

    return {key: results}


    

    
    

            
            
    
    
    