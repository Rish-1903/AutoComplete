import numpy as np
import pandas as pd
import math
import random
import nltk
nltk.data.clear_cache()  # Clear the cache
nltk.download('punkt')
nltk.data.path.append('.')
with open("en_US.twitter.txt", "r") as file:
    data = file.read()
def split_to_sentences(data):
    sentences=data.split("\n")
    sentences=[s.strip() for s in sentences]
    sentences=[s for s in sentences if len(s)>0]
    return sentences
def tokenize_sentences(sentences):
    tokenized_sentences=[]
    for sentence in sentences:
        sentence=sentence.lower()
        tokenized=nltk.tokenize.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)
    return tokenized_sentences

def get_tokenized_data(data):
    sentences=split_to_sentences(data)
    tokenized_sentences=tokenize_sentences(sentences)
    return tokenized_sentences

def count_words(tokenized_sentences):
    word_counts={}
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in word_counts.keys():
                word_counts[token]=1
            else:
                word_counts[token]+=1
    return word_counts

def get_words_with_nplus_frequency(tokenized_sentences,count_threshold):
    closed_vocab=[]
    word_counts=count_words(tokenized_sentences)
    for word,cnt in word_counts.items():
        if cnt>=count_threshold:
            closed_vocab.append(word)
    return closed_vocab
def replace_oov_words_by_unk(tokenized_sentences,vocabulary,unknown_token="<unk>"):
    vocabulary=set(vocabulary)
    replaced_tokenized_sentence=[]
    for sentence in tokenized_sentences:
        replaced_sentence=[]
        for token in sentence:
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)
        replaced_tokenized_sentence.append(replaced_sentence)
    return replaced_tokenized_sentence

def preprocess_data(train_data, test_data, count_threshold, unknown_token="<unk>",
                    get_words_with_nplus_frequency=get_words_with_nplus_frequency, replace_oov_words_by_unk=replace_oov_words_by_unk):
    vocabulary=get_words_with_nplus_frequency(train_data,count_threshold)
    train_data_replaced=replace_oov_words_by_unk(train_data,vocabulary,unknown_token)
    test_data_replaced=replace_oov_words_by_unk(test_data,vocabulary,unknown_token)
    return train_data_replaced,test_data_replaced,vocabulary




