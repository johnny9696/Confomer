import os
import sys
from symbols import symbols

def english_cleaner(text):
    lower_text=text.lower()
    lower_text=lower_text.replace('/"','')
    for i in lower_text:
        if i not in symbols:
            lower_text=lower_text.replace(i,'')
    return lower_text
