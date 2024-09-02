# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:22:56 2024

@author: Lenovo
"""

""" Datatypes 
1. Construct 2 lists containing all the available data types  (integer, float, string, complex and Boolean) and do the following..
a.	Create another list by concatenating above 2 lists
b.	Find the frequency of each element in the concatenated list.
c.	Print the list in reverse order."""


list1 = [123,12.3,"vishva", 5+7j, True]
list2 = [123,10.2,"mahesh", 2+7j, False]
list3 = list1 + list2
print(list3)

from collections import Counter
# Use Counter to find the frequency of each element
element_frequency = Counter(list3)

# Print the result
for element, frequency in element_frequency.items():
    print(f"{element}: {frequency}")

list3.reverse()
print(list3)


"""2.	Create 2 Sets containing integers (numbers from 1 to 10 in one set and 5 to 15 in other set)
a.	Find the common elements in above 2 Sets.
b.	Find the elements that are not common.
c.	Remove element 7 from both the Sets.
"""
set1 = set(range(1, 11))   # Numbers from 1 to 10
set2 = set(range(5, 16))   # Numbers from 5 to 15
print(set1)
print(set2)
print(set1 ^ set2)
set1.remove(7)
print(set1)
set2.remove(7)
print(set2)

"""3.	Create a data dictionary of 5 states having state name as key and number of covid-19 cases as values.
a.	Print only state names from the dictionary.
b.	Update another country and its covid-19 cases in the dictionary.
"""

dict1 = {"TN": 123, "Karnantaka" : 550, "AP" :121, "UP": 732, "JK":220}
print(dict1)
print(dict1.keys())
dict1["Kerala"] = 200
print(dict1)


"""Operators
Please implement by using Python
1.	A. Write an equation which relates   399, 543 and 12345 
B.  “When I divide 5 with 3, I get 1. But when I divide -5 with 3, I get -2”—How would you justify it?
"""

a = 399 
b = 543 
c = 12345
print(a == c % b)
# Floor Division
5 // 3 
-5 // 3

"""
2.  a=5,b=3,c=10.. What will be the output of the following:
              A. a/=b
              B. c*=5  
"""
a = 5 
b = 3 
c = 10
a /= b
c *= 5  

print(a)
print(c)

"""       
3. A. How to check the presence of an alphabet ‘S’ in the word “Data Science” .
            B. How can you obtain 64 by using numbers 4 and 3 .
"""

print('S' in "Data Science")
print(4 ** 3)


"""Variables
Please implement by using Python
1.	What will be the output of the following (can/cannot):
a.	Age1=5
b.	5age=55

2.	What will be the output of following (can/cannot):
a.	Age_1=100
b.	age@1=100

3.	How can you delete variables in Python ?
"""
Age1 = 5 #valid
5age = 55 #invalid decimal literal

Age_1 = 100 #valid
age@1 = 100 #invalid cannot assign to expression here

del Age_1
print(Age_1)

