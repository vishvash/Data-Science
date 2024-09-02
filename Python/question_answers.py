# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:22:05 2024

@author: Lenovo
"""

#1)Write a Python function that takes a string as input and converts it into a list of characters.

a = "Hello world"
b = [i for i in a]
print(b)

#2)Discuss the difference between a module and a package in Python. Provide examples of each and explain their respective use cases.
"""
Module: 
    Module is used to break down large programs into smaller, manageable parts.
    Module is a combination of reusable classes, functions, variables and codes.
    Built-in and custom modules can be used.
================================= module
mesge.py

def hi()
    print("hi how are you")
================================= code
import mesge

mesge.hi() \\output: hi how are you
=================================
Package: 
    Package is a combination of several related modules which has reusable classes, functions, variables and codes.
    Built-in and custom package can be used.
    It includes a special __init__.py file

================================= package
conversation/
├── __init__.py
├── mesge1.py
└── mesge2.py

mesge1.py

def hii(name)
    print(f"hi {name} how are you")

mesge2.py

def hello(name)
    print(f"hello {name} how are you")

================================= code    
from conversation import mesge1, mesge2

mesge1.hi("Mahesh") \\output: hi Mahesh how are you
mesge2.hello("Vishva") \\output: hello Vishva how are you
=================================
"""

#4)Create a function that receives a list of names and a specific name as input, and returns a boolean indicating whether the specific name is present in the list.

def checkname(name, *names):
    if name in names:
        return True
    else:
        return False
    
checkname("Vishva", "Swathi", "Deepak", "Vishva")

#5) What are Python Keywords?

"""
Keywords are the predefined words which cannot be used as variables.
Eg: if else break
"""

#6)What is a lambda function in Python and where is it useful?
"""
lamda function is an anonymous function used for simple operations
we don't need to define it with def keyword
"""
square = lambda x: x ** 2
print(square(5))#25

numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x ** 2, numbers))
print(squares) #[1, 4, 9, 16, 25]

# 7)What distinguishes the Python '==' and 'is' operators?
"""
If we compare two operands using == operator, it compares only the values and not whether they are the same object in the memory.
If we compare two operands using is operator, it compares only the values and not whether they are the same object in the memory.

"""
a = 5.4
b = 5.4

print(a == b) #True
print(a is b) #False

# 8) Write a Python function that takes a string as input and returns the reversed version of that string. For example, if the input is "hello," the output should be "olleh."


def reversed(name):
    return name[:: -1]

print(reversed("Vishva"))

#9)Create a Python list comprehension that generates a list of the squares of even numbers from 1 to 10.

a = [ i**2 for i in range (1,11)]

print(a) 

#10)Write a Python function to calculate the factorial of a given positive integer. The factorial of a number is the product of all positive integers up to that number. For example, the factorial of 5 (denoted as 5!) is 5*4*3*2*1.

def fact(n):
    if n < 0 :
        print("Please enter the valid number")
    elif n == 0 :
        return 1
    else:
        return(n*fact(n-1))
    
fact(5)
