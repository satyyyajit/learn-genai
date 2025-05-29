from nltk.tokenize import RegexpTokenizer

# tokenization - The process of breaking down text into smaller units, such as words or sentences, to facilitate analysis and processing.
# Tokenization is a fundamental step in natural language processing (NLP) that allows for the analysis of text data by breaking it down into manageable pieces.
tokenizer = RegexpTokenizer(r'\w+')  # Initialize tokenizer with regex pattern
text = "I like ibm"
tokens = tokenizer.tokenize(text)
print(tokens)  # Output: ['I', 'like', 'ibm']

# what is stop words removal?
# Stop words removal is the process of filtering out common words in a language that do not carry significant meaning, such as "the", "is", "in", etc. helps to reduce noise in text data and improve the performance of natural language processing tasks.
# decreases the size of the dataset, which can lead to faster processing and reduced memory usage.
# Example of stop words removal using NLTK's stopwords corpus

from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
# print(tokens) 
print('\nthe stopword removal process')

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print(filtered_tokens)


print('senthursiva.ibm@gmail.com\n')

# 3 Stemming
# converting the words to its root form 
# stemming is the process of reducing words to their base or root form, such as converting "running" to "run".
# Stemming is often used in information retrieval and text mining to improve search results and reduce dimensionality of text data.
# Example of stemming using NLTK's PorterStemmer

from nltk.stem import PorterStemmer

ps = PorterStemmer()
words = ['playing', 'gaming']

stems = [ps.stem(word) for word in words]
print('process of stemming :\n',stems)


# Lemmatization - Convert words into its dictionary form
# Lemmatization is the process of reducing words to their base or dictionary form, such as converting "better" to "good".
# for example -> better is lemmatized to good
# Lemmatization is often used in natural language processing tasks to improve the accuracy of text analysis and understanding.

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') # Download WordNet data for lemmatization
nltk.download('omw-1.4')  # Download Open Multilingual WordNet data for lemmatization
words = ['better', 'running', 'geese', 'mice']
lemmatizer = WordNetLemmatizer()

lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print('process of lemmatization :\n', lemmatized_words)
