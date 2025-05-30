from nltk.tokenize import TreebankWordDetokenizer
from nltk import pos_tag, word_tokenize
nltk.download('averaged_perceptron_tagger_eng')  # Download POS tagger data

sentence = "The cat is running"
tokens = tokenizer.tokenize(sentence)
pos_tags = pos_tag(tokens)
print('POS tagging of the sentence "The cat is running":\n', pos_tags)