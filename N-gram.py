import nltk
import math
import random
import numpy as np
import pandas as pd
from Preprocess import *
nltk.download('punkt')
with open("./en_US.twitter.txt","r") as f:
    data=f.read()

tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]
minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, 
                                                                        test_data, 
                                                                        minimum_freq)

def count_n_grams(data,n,start_token="<s>",end_token="<e>"):
    n_grams={}
    for sentence in data:
        sentence=[start_token]*n+sentence+[end_token]
        sentence=tuple(sentence)
        for i in range(len(sentence)-n+1):
            n_gram=sentence[i:i+n]
            if n_gram in n_grams.keys():
                n_grams[n_gram]+=1
            else:
                n_grams[n_gram]=1
    return n_grams

def estimate_probability(word,previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k=1.0):
    previous_n_gram=tuple(previous_n_gram)
    previous_n_gram_count=n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0
    denominator=previous_n_gram_count+k*vocabulary_size
    n_plus1_gram=previous_n_gram+(word,)
    n_plus_1_gram_count=n_plus1_gram_counts[n_plus1_gram] if n_plus1_gram in n_plus1_gram_counts else 0
    numerator=n_plus_1_gram_count+k
    probability=numerator/denominator
    return probability

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>",  k=1.0):
    previous_n_gram=tuple(previous_n_gram)
    vocabulary=vocabulary+[end_token,unknown_token]
    vocabulary_size=len(vocabulary)
    probabilities={}
    for word in vocabulary:
        probability=estimate_probability(word,previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k=k)
        probabilities[word]=probability
    return probabilities


def make_count_matrix(n_plus1_gram_counts, vocabulary):
    vocabulary = vocabulary + ["<e>", "<unk>"]
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]        
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    
    # mapping from n-gram to row
    row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}    
    # mapping from next word to column
    col_index = {word:j for j, word in enumerate(vocabulary)}    
    
    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix

def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix

def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>', end_token = '<e>', k=1.0):
    n=len(list(n_gram_counts.keys())[0])
    sentence=[start_token]*n+sentence+[end_token]
    sentence=tuple(sentence)
    N=len(sentence)
    product_pi=1.0
    for t in range(n,N):
        n_gram=sentence[t-n:t]
        word=sentence[t]
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)
        product_pi*=1/probability
    perplexity=(product_pi)**(1/N)
    return perplexity

def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>", k=1.0, start_with=None):
    n=len(list(n_gram_counts.keys())[0])
    previous_n_gram=previous_tokens[-n:]
    probabilities=estimate_probabilities(previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary,k=k)
    suggestion=None
    max_prob=0
    for word,prob in probabilities.items():
        if start_with:
            if not word.startswith(start_with):
                continue
        if prob>max_prob:
            suggestion=word
            max_prob=prob
    return suggestion,max_prob         

def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions


def get_user_input():
    user_input = input("Enter a sequence of words: ")
    previous_tokens = user_input.split()  # Split the input string into individual tokens (words)
    return previous_tokens

def suggest_and_complete_sentence(n_gram_counts_list, vocabulary, k=1.0):
    previous_tokens = get_user_input()
    
    start_with = input("Optional: Enter a starting letter or sequence for the suggested word (or press Enter to skip): ")
    start_with = start_with if start_with != "" else None
    
    # Get word suggestions
    suggestions = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=k, start_with=start_with)
    
    # Find the suggestion with the highest probability
    if suggestions:
        best_suggestion, max_prob = max(suggestions, key=lambda x: x[1])
        if best_suggestion:
            # Append the best suggestion to the previous tokens
            completed_sentence = previous_tokens + [best_suggestion]
            print("\nCompleted Sentence:")
            print(" ".join(completed_sentence))
        else:
            print("No suitable suggestion found.")
    else:
        print("No suggestions available.")

suggest_and_complete_sentence(n_gram_counts_list, vocabulary)


