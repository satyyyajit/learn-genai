
# tokenize

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

text = " I bought a new car from tata and I like it very much. It is a great car."   

tokens = tokenizer.tokenize(text)

print("original text:")
print(text)
print("tokenized text:")
print(tokens)

# stopwords removal

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')  

stop_words = set(stopwords.words('english'))
removed_stopwords = [word for word in tokens if word.lower() not in stop_words]

print('Stop words removed from the tokens:\n', removed_stopwords)

# stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()

stemmed_words = [ps.stem(word) for word in tokens]
print('Stemmed words from the tokens: ', stemmed_words)

# lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') 
nltk.download('omw-1.4') 

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

print('Lemmatized words from the tokens: ', lemmatized_words)

# pos tagging

from nltk.tokenize import TreebankWordDetokenizer
import nltk

nltk.download('averaged_perceptron_tagger_eng')

pos_tags = nltk.pos_tag(tokens)
print('POS tags for the tokens:')
print(pos_tags)

# named entity recognition 

import spacy
nlp = spacy.load("en_core_web_sm")

sentence = "I bought a new car from TATA and I like it very much. It is a great car."
text = nlp(sentence)

print("Named Entities, Phrases, and Concepts:")
for entity in text.ents:
    print(f"{entity.text} ({entity.label_})")