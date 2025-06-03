# text representation
# 1 Bag of Words (BoW) - A simple representation of text where each word is treated as a feature, and the frequency of each word is counted.
# 2 TF-IDF (Term Frequency-Inverse Document Frequency) - A more advanced representation that considers the importance of each word in a document relative to its frequency across multiple documents.
# 3 Word Embeddings - A dense vector representation of words that captures semantic relationships between words, such as Word2Vec or GloVe.




# 1 Bag of Words (BoW)

# i like to play football, football is great.
# i am going to feed both sentences to the BoW model
# it will convert each word into vector representation
# checks the unique words in the sentences and converts them into a vector

# for the sentence "i like to play football." it # will convert it into a vector like this
# [1, 1, 1, 0, 0]

# for the sentence "football is great." it will convert it into a vector like this
# [0, 0, 1, 1, 1]

# it will delete the duplicate words and create a vocabulary
# i like to play football is great

# for [1,1,1,0,0] it will convert it into a vector like this
# [1, 1, 1, 0, 0] -> i like to play football is great

from sklearn.feature_extraction.text import CountVectorizer 

texts = [
    "I like to play football.",
    "Football is great."
]

vectorizer = CountVectorizer()
# Fit the model and transform the texts into a bag-of-words representation

X = vectorizer.fit_transform(texts)

# Convert the sparse matrix to a dense format and get the feature names
print(vectorizer.get_feature_names_out(texts))
print(X.toarray())
# Output:
# ['football' 'great' 'i' 'like' 'play' 'to']


# 2 TF-IDF (Term Frequency-Inverse Document Frequency)
# it assigns a value to each word according to importance
# refined version of BoW
# it reduce the weight of common words and increase the weight of rare words
# like i, is , to, the, etc. are common words and they will have low weight
# while words like football, great, play will have high weight
# it is used to find the importance of a word in a document
# if a word appears frequently in a document but not in other documents, it will not be considered as a important word
# if a word appears less frequently in a document but appears in other documents, it will be considered as a important word

from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "I like to play football.",
    "Football is great."
]

tf_idf = TfidfVectorizer()

# Fit the model and transform the texts into a TF-IDF representation
X = tf_idf.fit_transform(texts)

# Convert the sparse matrix to a dense format and get the feature names
print(tf_idf.get_feature_names_out(texts))
print(X.toarray())
# Output:
# ['football' 'great' 'i' 'like' 'play' 'to']
# [0.57735027 0.57735027 0.57735027 0.         0.         0.        ]
# [0.         0.57735027 0.         0.         0.         0.57735027]

# word embeddings
# 3 Word Embeddings
# Word embeddings are dense vector representations of words that capture semantic relationships between them.
# what is semantic relationships?
# Semantic relationships refer to the meanings and associations between words, such as synonyms, antonyms, and related concepts.
# Word embeddings are used to represent words in a continuous vector space, where similar words are closer together.
# Example of word embeddings -
# I like to play football.
# In word embeddings, the words "football" and "play" would be represented as vectors in a high-dimensional space, where their positions reflect their meanings and relationships.
# we use pre-trained word embeddings like Word2Vec, GloVe, or FastText to represent words in a continuous vector space.
# HERE WE USE Gensim's Word2Vec to create word embeddings

