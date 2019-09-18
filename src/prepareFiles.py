# removes puncutation, stems original documents,
# and saves as a csv, with ranking seperated by ':'
#===============imports==================#
import nltk
import re
import pickle
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import *
#==============functions==================#
def prepareFile():
    # Open the training file 
    output = open("train_file_trimmed.txt", "w")
    # Initialize stemmer
    stemmer = PorterStemmer()
    # Initialize Stop Words
    stop_words = set(stopwords.words('english')) 
    # Open the file, loop through
    with open("train_file.dat") as f:
        for line in f:
            #gets arting
            rating = line[0:2]
            docs_rating.append(rating)
            line = line[3:]
            #processes line (stems, removes punctuation)
            line = " ".join(re.findall(r"[a-zA-Z]+", line))
            word_tokens = word_tokenize(line) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            stemmed_sentence = [stemmer.stem(word) for word in filtered_sentence]
            final_string = rating+':'
            final_string = final_string+",".join(stemmed_sentence)
            output.write(final_string+"\n")
    output.close()

def main():
    prepareFile()
main()
