# takes in dimension reduced test weights, and iterates through
# the test documents, searching for their K nearest neighbors in
# a K-D Tree built from the training data. Determines the category
# of the test document based on the ratings of the KNN
#=======================imports===========================#
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree
import time
#=======================globals=========================#
START_TIME= time.time()
K_VAL = 7
#=====================functions=======================#
def categorize():
    # loads necessary info
    test_data = pickle.load(open('./pickles/sparse_word_weight_test.p','rb'))
    train_data = pickle.load(open('./pickles/sparse_docs_weight_reduced.p','rb'))
    ranking = pickle.load(open('./pickles/docs_rating.p','rb'))
    #builds K-D Tree based on training data
    kdt = KDTree(train_data.toarray(), leaf_size=30, metric='euclidean')
    output = open("test_ranks.txt", "w")
    count =0
    #goes through reduced test weights finds knn 
    # and categorizing the documents
    for line in test_data.toarray():
        if(count % 100 == 0):
            print(count)
        #queries k-d tree for k nearest neighbors of document
        line = line.reshape(1,-1)
        dist,ind = kdt.query(line,k=K_VAL,return_distance=True,dualtree=True)
        rating = 0
        #finds ranking based on knn's
        for val in range(K_VAL):
            rating += int(ranking[ind[0][val]]) * (1/(dist[0][val]**2))
        if(rating >=0):
            rating = "+1"
        else:
            rating = "-1"
        output.write(rating+"\n")
        count+=1
    output.close()
    print("--- %s seconds ---" % (time.time() - START_TIME))
def main():
    categorize()
main()