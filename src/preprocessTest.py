# Calculates normalized weights of test data, stores in a 
# sparse matrix, and runs a dimension reduction on the sparse matrix
#=====================imports====================#
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import *
import re
from math import log10
from sklearn.decomposition import TruncatedSVD
import pickle
from scipy.sparse import csr_matrix
#===================GLOBAL=======================#
TEST_DATA = 'test.dat'
DIMENSIONS = 100
#===================functions====================#
def main():
    keys_dict = pickle.load(open('./pickles/keys_dict.p','rb')) 
    keys_list = pickle.load(open('./pickles/keys_list.p','rb')) 
    svd_transform = pickle.load(open('./pickles/transform.p','rb'))
    docs_containing_word_count = pickle.load(open('./pickles/docs_containing_word_count.p','rb'))
    stemmer = PorterStemmer()
    # Initialize Stop Words
    stop_words = set(stopwords.words('english')) 
    line_word_weight = []
    with open(TEST_DATA) as f:
        count = 0
        for line in f:
            #stems,removes punctuation, and counts occurances
            #for all test documents
            if(count %100 == 0):
                print(count)
            temp_ar = [0] * len(docs_containing_word_count.keys())
            line = " ".join(re.findall(r"[a-zA-Z]+", line))
            word_tokens = word_tokenize(line) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            stemmed_sentence = [stemmer.stem(word) for word in filtered_sentence]
            for word in set(stemmed_sentence):
                #count words
                if(word in keys_list):
                    temp_ar[keys_dict[word]] += 1
            divisor = 0
            #sets non normalized weight of terms in document
            for val in range(len(temp_ar)):
                log_value = log10(len(docs_containing_word_count.keys())/docs_containing_word_count[keys_list[val]])
                divisor += temp_ar[val]**2 * log_value **2
                temp_ar[val] = temp_ar[val] * log_value
            #normalizes weights
            for val in range(len(temp_ar)):
                if(divisor <=0):
                    divisor = 1
                temp_ar[val] = temp_ar[val]/(divisor**(.5))
            line_word_weight.append(temp_ar)
            count+=1
    #saves weights in sparse matrix
    sparse_word_weight_test = csr_matrix(line_word_weight)
    #runs svd on test weights
    svd = TruncatedSVD(n_components=DIMENSIONS, n_iter=7, random_state=42)
    sparse_word_weight_test = csr_matrix(svd_transform.transform(sparse_word_weight_test))
    #saves dimension reduced matrix
    pickle.dump(sparse_word_weight_test,open('./pickles/sparse_word_weight_test.p','wb'))
main()