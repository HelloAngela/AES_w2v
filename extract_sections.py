# read txt into one string
# regex to match the start and end of a substring

import re
import glob
import pandas as pd


filedir = "C:/Users/hello/Desktop/MIMI2002/all/*.txt"
txt_1d = []
sid_1d = []
sid_1d_error = []
# q2a = re.compile(r"Question A \(4 marks, ~10 words\)", re.IGNORECASE)
# q2b = re.compile(r"Question B \(6 marks, ~50 words\)", re.IGNORECASE)
# txt_2a = []
# txt_2b = []
# sid_2a = []
# sid_2a_error = []

# raw text: Question A (4 marks, ~10 words)
# raw text: Question B (6 marks, ~50 words)
# raw text: Question C (10 marks, ~100 words) (can't be used since there are two versions)
# raw text: Question D (6 marks, ~60 words)
# raw text: Question E (2 marks, ~20 words)
# raw text: Scenario 3 Answers

for file in glob.glob(filedir):
    with open(file, encoding="utf8") as f:
        text = f.read().replace('\n', '')
        sid = file[-13:-4]

        # for scenario A, question D
        s = re.split(r"Question D \(6 marks, ~60 words\)"
                     r"|Question E \(2 marks, ~20 words\)", text)
        try:
            txt_1d.append(s[1])
        except IndexError:
            sid_1d_error.append(sid)
            print("Index error with" + " " + sid)
        else:
            sid_1d.append(sid)

        # for Scenario B, Questions A B and C
        # s = re.split(r"Question A \(4 marks, ~10 words\)"
        #              r"|Question B \(6 marks, ~50 words\)"
        #              r"|Question C \(10 marks, ~100 words\)|Scenario 3 Answers", text)
        #
        # try:
        #     txt_2a.append(s[1])
        # except IndexError:
        #     sid_2a_error.append(sid)
        #     print("Index error with" + " " + sid)
        # else:
        #     txt_2b.append(s[2])
        #     sid_2a.append(sid)

df = pd.DataFrame()
df['SID'] = sid_1d
df['txt_1d'] = txt_1d
df_marks = pd.read_csv("C:/Users/hello/Desktop/MIMI2002/marks_scene1d.csv")
df_marks2 = df_marks.dropna(subset='SID')
df_marks2 = df_marks2.astype({'SID': 'int32'})
df = df.astype({'SID': 'int32'})
df_combined = df.merge(df_marks2, on='SID')
df_combined['score_1d']= df_combined['score_1d']*2
df_combined = df_combined.astype({'score_1d': 'int32'})
# multiplication to get rid of 0.5marks, easier to train with
# df_combined['score_2a']=df_combined['score_2a']/2
# df_combined['score_2b']=df_combined['score_2b']*2


df_combined.to_csv("C:/Users/hello/Desktop/MIMI2002/mimi_scene1d.csv", sep='\t', index=False)


# df = pd.DataFrame()
# df['SID'] = sid_2a
# df['txt_2a'] = txt_2a
# df['txt_2b'] = txt_2b
# df_marks = pd.read_csv("C:/Users/hello/Desktop/MIMI2002/marks_sec2.csv")
# df_marks2 = df_marks.dropna(subset='SID')
# df_marks2 = df_marks2.astype({'SID': 'int32'})
# df = df.astype({'SID': 'int32'})
# df_combined = df.merge(df_marks2, on='SID')
# df_combined['score_2b']= df_combined['score_2b']*2
# df_combined = df_combined.astype({'score_2a': 'int32'})
# df_combined = df_combined.astype({'score_2b': 'int32'})
# # multiplication to get rid of 0.5marks, easier to train with
# # df_combined['score_2a']=df_combined['score_2a']/2
# # df_combined['score_2b']=df_combined['score_2b']*2
#
#
# df_combined.to_csv("C:/Users/hello/Desktop/MIMI2002/mimi_scene2.csv", sep='\t', index=False)

