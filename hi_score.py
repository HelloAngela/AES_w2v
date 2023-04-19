import pandas as pd

pd.set_option('display.max_colwidth', None)

f = open("C:/Users/hello/Desktop/2018 paper/txt_as_list3.csv", "r", encoding="utf-8")
txt = pd.read_csv(f)
f.close()
hiscore = txt[txt.score > 14]
l_txt = hiscore['essay']

with open("C:/Users/hello/Desktop/AES_WE/report_corpus.txt", "w", encoding="utf-8") as c:
    cw = l_txt.to_string(header=False, index=False)
    c.write(cw)
