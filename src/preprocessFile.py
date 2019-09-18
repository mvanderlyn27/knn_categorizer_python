# calculates weights for all documents in the 
# trial data and stores in sparse array  
#=====================imports===================#
import nltk
import re
import pickle
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import *
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix
from math import log10
import numpy as np
#===================globals=========================#
FILENAME = 'train_file_trimmed.txt'
#===================files=========================#
def processFile():
    # Open the file as read
    output = open("train_file_processed.txt", "w")
    # Initialize stemmer
    stemmer = PorterStemmer()
    # Initialize Stop Words
    stop_words = set(stopwords.words('english')) 
    docs_containing_word_count = {}
    docs_rating  = []
    line_word_weight = [] # 2d array size = # of unique words (for holding weight)
    # Open the file find # words, and word count
    count = 0
    with open(FILENAME) as f:
        for line in f:
            if(count %100 == 0):
                print(count)
            docs_rating.append(line[0:2])
            #gets docs containing word count 
            for word in set(line[3:].strip().split(',')):
                if(word in docs_containing_word_count.keys()):
                    docs_containing_word_count[word] +=1
                else:
                    docs_containing_word_count[word] = 1
            count+=1
    print("gathered unique words")
    #maps keys to index of key in array
    keys_list = list(docs_containing_word_count.keys())
    keys = {}
    count = 0
    for key in keys_list:
        keys[key] = count
        count+=1
    #calc normalized weights
    with open(FILENAME) as f:
        count = 0
        for line in f:
            if(count %100 == 0):
                print(count)
            temp_ar = [0] * len(docs_containing_word_count.keys())
            for word in set(line[3:].strip().split(',')):
                #count words
                temp_ar[keys[word]] += 1
            divisor = 0
            for val in range(len(temp_ar)):
                #sets non normalized weight of 
                log_value = log10(len(docs_containing_word_count.keys())/docs_containing_word_count[keys_list[val]])
                divisor += temp_ar[val]**2 * log_value **2
                temp_ar[val] = temp_ar[val] * log_value
            for val in range(len(temp_ar)):
                temp_ar[val] = temp_ar[val]/(divisor**(.5))
            line_word_weight.append(temp_ar)
            count+=1
            #calculate values based off term frequency
    print("done with weights, demension reduction")
    #turns weights into sparse matrix
    sparse_word_weight = csr_matrix(line_word_weight)
    #saves info with pickle for use later
    pickle.dump(sparse_word_weight,open('./pickles/sparse_word_weight.p','wb'))   
    pickle.dump(docs_rating, open('./pickles/docs_rating.p','wb'))
    pickle.dump(keys_list, open('./pickles/keys_list.p','wb'))
    pickle.dump(keys, open('./pickles/keys_dict.p','wb'))
    pickle.dump(keys_list, open('./pickles/keys_list.p','wb'))
    pickle.dump(docs_containing_word_count,open('./pickles/docs_containing_word_count.p','wb'))
def main():
    processFile()
main()


