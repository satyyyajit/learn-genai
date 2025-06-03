# What is Transformers Library?
# A Library for NLP 
# Hugging Face Transformers is a library that provides pre-trained models 
# It includes models for tasks such as text classification, named entity recognition, question answering, and more. 
# The library is built on top of PyTorch and TensorFlow, making it easy to use with both frameworks.

# word -> token -> embedding vector

# input embedding 
# convert words to tokens into embedding vectors
# they capture semantic relationships between words
# canm be learned from scratch or pre-trained on large datasets
# Word2Vec, GloVe, FastText are popular word embedding techniques


# positional encoding
# transforemer does not hjave recurrent connections
# to preserve the order of words in a sequence, positional encoding is added to the input embeddings.
# these encodings follow fixed or learned patterns to indicate the position of each token in the sequence.

# Multi-head self-attention
# allow each token to attend to all other tokens in the sequence
# capture different types of relationships between tokens
# uses a mechanism of softmax to compute attention scores
# feed forward neural networks
# each token's attention output is passed through a feed-forward neural network
# Linear-> ReLU -> Linear

# SStacked layers
# Transformers consist of multiple stacked layers, each containing multi-head self-attention and feed-forward neural networks.
# lot of hidden layers
# This allows the model to learn complex relationships and representations of the input data.
# DEEP LAYERS -> BEtter REPRESENTATION

#Output layer
# The output layer typically consists of a linear layer followed by a softmax activation function for classification tasks.

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

ids = tokenizer('It was a dark and stormy', return_tensors='pt').input_ids
print(ids)

for id in ids[0]:
    print(id, '\t:', tokenizer.decode(id))