# reduces dimensions of training data using truncated
# SVD
#====================imports===================#
from sklearn.decomposition import TruncatedSVD
import pickle
from scipy.sparse import csr_matrix
#====================globals=======================#
DIMENSIONS = 100
#====================functions=====================#
def main():
    # loads necessary data
    rankings = pickle.load(open('./pickles/docs_rating.p','rb'))
    sparse_word_weight = pickle.load(open('./pickles/sparse_word_weight.p','rb'))
    print("loaded data")
    # reduces dimensions to 100
    svd = TruncatedSVD(n_components=DIMENSIONS, n_iter=7, random_state=42) #LIKE PCA, but for Sparse data
    sparse_word_weight = csr_matrix(svd.fit_transform(sparse_word_weight))
    print(sparse_word_weight)
    print(len(sparse_word_weight.toarray()[0]))
    # saves reduced matrix, and transformation to be used on test data
    pickle.dump(sparse_word_weight, open('./pickles/sparse_docs_weight_reduced.p','wb'))
    pickle.dump(svd, open('transform.p','wb'))
main()
