# Using only a part of the txt for analysis

import pandas as pd

# with open("C:/Users/hello/Desktop/2018 paper/txt_as_list.csv", "r", encoding="utf-8") as f:
#     txt = pd.read_csv(f, delim_whitespace=True)

f = open("C:/Users/hello/Desktop/2018 paper/txt_as_list2.csv", "r", encoding="utf-8")
txt = pd.read_csv(f, sep=',')
f.close()

l_500 = []
l_1000 = []
l_1500 = []
ser = txt['essay']
# split essay into 500 word chunks
# a = txt.at[0, 'essay']

for index, value in ser.items():
    split = str(value).split(' ')
    f500 = split[1:501]
    f1000 = split[500:1001]
    f1500 = split[1000:1500]

    separator = ' '
    joined500 = separator.join(f500)
    joined1000 = separator.join(f1000)
    joined1500 = separator.join(f1500)

    l_500.append(joined500)
    l_1000.append(joined1000)
    l_1500.append(joined1500)


txt['essay_500'] = l_500
txt['essay_1000'] = l_1000
txt['essay_1500'] = l_1500
txt.to_csv("C:/Users/hello/Desktop/2018 paper/txt_as_list3.csv", encoding='utf-8', index=False)
