# Task 1 solution. It uses Training_set and Training_benchmark.txt

import math
from stemming.porter2 import stem

def avg_doc_len(coll):
    tot_dl = 0
    for id, doc in coll.get_docs().items():
        tot_dl = tot_dl + doc.get_doc_len()
    return tot_dl/coll.get_num_docs()
    
def w5(coll, ben, theta):
    # Find D_plus, The set of relevant documents
    D_plus = set()
    for docID, relevance_judgement in ben.items():
        if relevance_judgement == 1:
            D_plus.add(docID)

    N = coll.get_num_docs()
    # The size of all relevant documents
    R = len(D_plus)

    # All terms in D_plus
    T = set()
    for docID, doc in coll.get_docs().items():
        if docID in D_plus:
            # add terms to T
            T.update(doc.get_term_list())

    # step 2
    n = {}
    r = {}
    for tk in T:
        n[tk] = 0
        r[tk] = 0

    # step 3
    for tk in T:
        for doc in coll.get_docs().values():
            if tk in doc.get_term_list():
                n[tk] = n[tk] + 1

    # step 4
    docs = coll.get_docs()
    for tk in T:
        for docID in D_plus:
            doc = docs[docID]
            if tk in doc.get_term_list():
                r[tk] = r[tk] + 1

    # step 5
    W_5 = {}
    for tk in T:
        W_5[tk] =( (r[tk] + 0.5)/(R - r[tk] + 0.5) ) / ( (n[tk]-r[tk]+0.5)/((N-n[tk])-(R-r[tk])+0.5) )

    W_5_mean = 0
    for w5 in W_5.values():
        W_5_mean += w5
    W_5_mean /= len(T)

    return {tk:w5 for tk, w5 in W_5.items() if w5 > (W_5_mean + theta)}
if __name__ == "__main__":

    import sys
    import os
    import coll
    #import df

    if len(sys.argv) != 2:
        sys.stderr.write("USAGE: %s <coll-file>\n" % sys.argv[0])
        sys.exit()
    coll_fname = sys.argv[1]  # use Training_set as an input argument
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    # use document parsing to represent training set
    coll_ = coll.parse_rcv_coll(coll_fname, stop_words)
    
    os.chdir('..')
    benFile = open('Training_benchmark.txt') # the relevance judgements
    file_ = benFile.readlines()

    ben={}
    for line in file_:
        line = line.strip()
        lineList = line.split()
        ben[lineList[1]]=float(lineList[2])

    benFile.close()
    theta = 3.5
    bm25_weights = w5(coll_, ben, theta)
    
    
    wFile = open('Model_w5_R102.dat', 'w')
    for (k, v) in sorted(bm25_weights.items(), key=lambda x: x[1], reverse=True):
        wFile.write(k +' '+ str(v) +'\n')
    wFile.close()

            
