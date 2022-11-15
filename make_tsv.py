# Converting current data type to tsv so to be compatible with data_utils

import pandas as pd
import glob
import os
import csv

# generating a list of file names for iterative text conversion later
with open("C:/Users/hello/Desktop/2018 paper/list_txt.txt", "w") as ls:
    txt_list = glob.glob(pathname="C:/Users/hello/Desktop/2018 paper/Training_set_txt/*.txt")
    for lines in txt_list:
        ls.writelines(lines+'\n')
# load the list of file names to a panda dataframe
df = pd.DataFrame()
df = df.append(pd.DataFrame(txt_list), ignore_index=True)
df2 = pd.read_csv("C:/Users/hello/Desktop/2018 paper/analysis/train_grades.csv")
grade = df2['Grade']

# truncating the text file paths into paper ID for analysis later
df1 = df[0].str[-14:-4]

# To fit each essay into one excel cell, I converted the text into a list of words, and then joining
# all words in one list to create one string. This string is then inserted into a list (l_combo).

l_combo = []
for i in txt_list:
    # dftxt = pd.read_csv(os.path.abspath(i))
    with open(os.path.abspath(i), encoding='utf-8') as txtfile:
        print("processing "+i+'...')
        lines = [line.rstrip() for line in txtfile]
        separator = ' '
        combo = separator.join(lines)
        print("adding " + i + ' to text list')
        l_combo.append(combo)

df['essay'] = l_combo
df['essay_id'] = df1
df['score'] = grade
df.to_csv("C:/Users/hello/Desktop/2018 paper/txt_as_list.csv", encoding='utf-8', index=False)

