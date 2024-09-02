# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:12:40 2024

@author: Lenovo
"""

"""
Conditional Statements
Please write Python Programs for all the problems .
1.	 Take a variable ‘age’ which is of positive value and check the following:
a.	If age is less than 10, print “Children”.
b.	If age is more than 60 , print ‘senior citizens’
c.	 If it is in between 10 and 60, print ‘normal citizen’

2.	Find  the final train ticket price with the following conditions. 
a.	If male and sr.citizen, 70% of fare is applicable
b.	If female and sr.citizen, 50% of fare is applicable.
c.	If female and normal citizen, 70% of fare is applicable
d.	If male and normal citizen, 100% of fare is applicable
[Hint: First check for the gender, then calculate the fare based on age factor.. For both Male and Female ,consider them as sr.citizens if their age >=60]
3.	Check whether the given number is positive and divisible by 5 or not.  
"""


age = int(input("Enter your age: "))
if age < 10 :
    print("Children")
elif age > 60 :
    print("Senior Citizen")
elif age <= 60 and age >= 10 :
    print("Normal Citizen")
    
age = int(input("Enter your age: "))
gender = input("Enter your gender: ")

if gender == "male" and age > 60 :
    print("70% of fare is applicable")
elif gender == "female" and age > 60 :
    print("50% of fare is applicable")
elif gender == "female" and age <= 60 and age >= 10 :
    print("70% of fare is applicable")
elif gender == "male" and age <= 60 and age >= 10 :
    print("100% of fare is applicable")
    
num = int(input("Enter the number: "))

if num > 0 :
    if num % 5 == 0 :
        print("The given number is positive and divisible by 5")
    else :
        print("The given number is positive but not divisible by 5")
else :
    print("The given number is not positive")
    
"""
Control Statements
Please implement Python coding for all the problems.

1.	
A) list1=[1,5.5,(10+20j),’data science’].. Print default functions and parameters exist in list1.
B) How do we create a sequence of numbers in Python?
C)  Read the input from keyboard and print a sequence of numbers up to that number
"""

list1=[1,5.5,(10+20j),"data science"]

dir(list)
for i in dir(list1) :
    print(i)
print('\n')
for i in list1 :
    print(i)

a = list(range(1,16))
a
print(a)

a = int(input("Enter a number: "))
i = 1 
while i < a :
    print(i)
    i += 1
    
"""
2.	Create 2 lists.. one list contains 10 numbers (list1=[0,1,2,3....9]) and other 
list contains words of those 10 numbers (list2=['zero','one','two',.... ,'nine']).
 Create a dictionary such that list2 are keys and list 1 are values..

"""
a = list(range(10))
import inflect
# Create a list containing words for numbers 0 to 9 using inflect
p = inflect.engine()
b = [p.number_to_words(num) for num in range(10)]
dict1 = dict(zip(b,a))
print(a)
print(b)
print(dict1)

# import os
# import sys
# os.path.dirname(sys.executable)
'''

3.	Consider a list1 [3,4,5,6,7,8]. Create a new list2 such that Add 10 to the even number and multiply with 5 if it is an odd number in the list1..

4.  Write a simple user defined function that greets a person in such a way that :
             i) It should accept both the name of the person and message you want to deliver.
              ii) If no message is provided, it should greet a default message ‘How are you’
           Ex: Hello ---xxxx---, How are you  -🡪 default message.
            Ex: Hello ---xxxx---, --xx your message xx---\
'''

list1 = [3,4,5,6,7,8]
list2 = [(num + 10) if num % 2 == 0 else (num * 5) for num in list1]
print(list2)

def message(name, msg="How are you") :
    print("Hello {}, {}".format(name, msg))
    
nam = "mahesh"
mg = "All the best for your exam"
message(nam, mg)
message(nam)

