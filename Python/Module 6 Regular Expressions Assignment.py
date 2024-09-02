# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 08:49:25 2024

@author: Lenovo
"""

"""
Regular Expression
1)Write a Python program to check that a string contains only a certain set of characters (in this case a-z, A-Z and 0-9)	
2) Write a Python program to replace all occurrences of space, comma, or dot with a colon.
"""

import re

def allowed_chars(input_str):
    pattern = r'^[a-zA-Z0-9]+$'
    match = re.match(pattern, input_str)
    return bool(match)

test_string = input("Enter a set of characters: ")
result = allowed_chars(test_string)

if result:
    print(f"The string '{test_string}' contains only allowed characters.")
else:
    print(f"The string '{test_string}' contains other characters.")


def replace_with_colon(input_str):
    pattern = r'[ ,.]'
    replaced_str = re.sub(pattern, ':', input_str)
    return replaced_str

# Example usage:
original_string = input("Enter a string: ")
modified_string = replace_with_colon(original_string)

print(f"Original String: {original_string}")
print(f"Modified String: {modified_string}")
