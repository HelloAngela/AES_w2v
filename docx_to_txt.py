# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:07:02 2022
@author: hello
"""
import os
import re
import glob
import pypandoc


def convert_docx_to_txt_and_rename(docx_folder_path):
    """
    Converts all .docx files in the given folder to .txt files,
    and renames each file to a 9-digit number extracted from its
    original file name using regex.
    """
    # Find all .docx files in the folder
    docx_files = glob.glob(os.path.join(docx_folder_path, '*.docx'))

    # Loop over each file and convert to .txt and rename
    for docx_file in docx_files:
        # Convert .docx file to .txt
        txt_file_path = os.path.splitext(docx_file)[0] + '.txt'
        pypandoc.convert_file(docx_file, 'plain', outputfile=txt_file_path)

        # Rename .txt file to 9-digit number extracted from original file name
        match = re.search(r'\d{9}', os.path.basename(docx_file))
        if match:
            new_file_name = match.group() + '.txt'
            os.rename(txt_file_path, os.path.join(os.path.dirname(txt_file_path), new_file_name))
            print(f"{os.path.basename(docx_file)} converted to {new_file_name}")
        else:
            print(f"No 9-digit number found in {os.path.basename(docx_file)}")

# # Converting docx without subfilename to txt
# filedir = "C:/Users/hello/Desktop/MIMI2002/all/*"
# for filename in glob.glob(filedir):
#     outF = pypandoc.convert_file(filename, 'plain', outputfile=filename + '.txt', format='docx')

