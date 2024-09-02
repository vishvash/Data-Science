# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:08:13 2024

@author: Lenovo
"""

'''
File Handling & Exception Handling

1)Write the code in python to open a file named “try.txt”
2)What is the purpose of ‘r’ as a prefix in the given statement? 
     f = open(r, “d:\color\flower.txt”)
3)Write a note on the following
A.	Purpose of Exception Handling
B.	Try block
C.	Except block
D.	Else block
E.	Finally block
F.	Built-in exceptions
4) Write 2 Custom exceptions
'''

f = open("try.txt", "w")
f.close()

'''
2.
"r" - opens the file in read mode


3.
A. Purpose of Exception Handling:

Exception handling in Python facilitates graceful management of runtime errors, preventing program crashes and enabling recovery from unexpected situations.

B. Try block:

The "try" block encloses code where exceptions might occur, allowing for proactive handling of potential errors.

C. Except block:

The "except" block follows a "try" block, catching and handling specific exceptions, defining actions to be taken when errors occur.

D. Else block:

The "else" block, used with exception handling, contains code to execute when no exceptions occur in the preceding "try" block.

E. Finally block:

The "finally" block is employed for code that must execute whether an exception occurs or not, often used for cleanup operations.

F. Built-in exceptions:

Built-in exceptions in Python are predefined error classes, like TypeError and ValueError, representing specific types of errors that can be caught and handled in a program.
'''

x = int(input('Enter the postive number: '))
if x < 1:
  raise Exception("Sorry, enter a valid number")
else:
    print('you entered: ', x)
    
    
x = input('Enter the number: ')
if x.isnumeric():
    print('you entered: ', x)
else:
    raise Exception("Sorry, enter a valid number")
