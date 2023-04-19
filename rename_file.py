import re
import glob
import os


filedir = "C:/Users/hello/Desktop/MIMI2002/all/*.doc*"
pattern = re.compile(r"\d\d\d\d\d\d\d\d\d", re.IGNORECASE)

for filename in glob.glob(filedir):
    basename = os.path.basename(filename)
    sid_match = pattern.search(basename)
    sid = sid_match.group(0)
    os.rename(filename, sid)
