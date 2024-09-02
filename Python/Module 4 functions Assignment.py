# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 08:49:06 2024

@author: Lenovo
"""

'''
1.	 Write a Python function to find the Max of three numbers.
2.	 Write a Python function to sum all the numbers in a list.
3.	 Write a Python function to multiply all the numbers in a list.
'''

def maximum (num1, num2, num3) :
    if num1 >= num2 and num1 >= num3:
        return num1
    elif num2 >= num1 and num2 >= num3:
        return num2
    else:
        return num3
print(maximum(2,3,4))


from functools import reduce
def add(listofnos) :
    return reduce(lambda x, y: x + y, listofnos)
list1 = [1, 2, 3, 4, 5]
print(add(list1))

from functools import reduce
def multiply(listofnos) :
    return reduce(lambda x, y: x * y, listofnos)
list1 = [1, 2, 3, 4, 5]
print(multiply(list1))

 