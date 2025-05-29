from nltk.tokenize import *

def regex_tokenizer(text:str)->list[str]:
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)