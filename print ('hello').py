first_number = 5
second_number = 7
result = first_number + second_number
print(f"{first_number} + {second_number} = {result}")
second_number = 4
print (second_number)
import pandas as pd
dat = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(dat)
print(df)   
#pip install numpy
import numpy as np
#pip install pandas matplotlib
import matplotlib.pyplot as plt
conda list
pd.__version__
np.__version__
plt.__version__
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
from helper.tools import a
students = 'Robert'
print (students)
students.type
hello = 'test'
gc.collect(hello)
print(hello)
d = "First line.\nSecond line."
from decimal import Decimal
input = Decimal('0.1') + Decimal('0.2')
print(input)
#Two ways to write comments in python
"""
#wrtie a function that prints hello
Comments can also be written like this
"""
# Sequence Types in python
my_list = [1, 2, 3, 4, 5] #python is zero indexed
my_list [0]
my_list.insert(2, 6) #inserts 6 at index 2
print (my_list)
my_list.remove (4) #removes first occurrence of 4



#not very frequently used, but in deep learing models, tumples are immutabel, i.e. intems are fixed after assignment
tup = (1, 2, 3, 4, 5)
#convert to list, modify and convert back to tuple
# 
range_name = range(0, 20, 2) #start, stop, step
print (range_name)
for year in range (2000, 2023):
    print (year)

#dictionary
#standing mapping type in python, indexed by keys 
d = {"one": 1, "two": 2, "three": 3}
list(d.keys())
list(d.values)

#Lists: For ordered mutalble colleciton of items, especially when itema might be added, ereased, or modified
#Tuples: For ordered immutable collection of items, especially when the collection should not be modified
#Ranges: For ordered immutable collection of numbers, especially for looping a specific number of times in
#Dictionaries: For unordered mutable collection of key-value pairs, especially when fast lookups by key are needed
#set: For unordered mutable collection of unique items, especially when membership testing and eliminating duplicates are important

print('Hello Crops!')

crop = {'wheat', 'corn', 'rice'}

print (f'Hello Crops! {crop}')

for index, value in enumerate(crop):
    print (value)# f string
    print (f"{value} is the {index} position in the set and its value is {value}")

info = {'name': Paul, 
 'Gender': Male, 
 'Uni ': Uni, 
 'IBAN': 1234567890}

for key, value in info.items():
    print (key, value)

# Entwas austauschen, Wort auswählen, dann Ctrl + D to select next occurence

#logging
import logging

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
    logging.FileHandler("experiment.log"),
    logging.StreamHandler()
]
)


logging.info("This is an info message")
#Fortschrittsbalken mit tqdm
pip install tqdm
from tqdm import tqdm
import time

for epoch in tqdm(range(20), desc = 'Processing..'):
    print('Start process')
    time.sleep (10)

#Import files package to see the files on mac
import os 
files = os.listdir('/Users/paulkonopka/Documents/Uni')
print (files)

#different way to do the same thing
from pathlib import Path
p = Path('/Users/paulkonopka/Documents/Uni')

from pathlib import Path
files = list(Path("/Users/paulkonopka/Documents/Uni").iterdir())
print(files)
folder = Path("images")
x = 5
y = 120