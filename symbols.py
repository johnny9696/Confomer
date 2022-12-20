import os
import sys

_alphabet='abcdefghijklmnopqrstuvwxyz'
_num='0123456789'
_symbol=".,;:!?'`\" "
_special="_^&/-"

symbols=['<BNK>']+list(_alphabet)

def text2vec(text):
    result=[]
    text=list(text)
    for i in text:
        if i in symbols:
            result.append(symbols.index(i))
        else:
                raise Exception("text {} is not in symbol list".format(i))
        
    return result

def vec2text(vec):
    result=[]
    for num,i in enumerate(vec):
        if i == len(symbols):
            continue
        else:
            try:
                if vec[num-1]!=i:
                    result.append(symbols[i])  
            except :
                raise Exception("There is not {} in list".format(i))
    result=''.join(result)
    return result