# It updates week9 Task1 solution.
# It uses Training_set and PTraining_benchmark.txt

import math
from stemming.porter2 import stem

def avg_doc_len(coll):
    tot_dl = 0
    for id, doc in coll.get_docs().items():
        tot_dl = tot_dl + doc.get_doc_len()
    return tot_dl/coll.get_num_docs()
    
def w5(coll, ben, theta):
    T={}
    # select T from positive documents and r(tk)
    for id, doc in coll.get_docs().items():
        if ben[id] > 0:
            for term, freq in doc.terms.items():
                try:
                    T[term] += 1
                except KeyError:
                    T[term] = 1
    #calculate n(tk)
    ntk = {}
    for id, doc in coll.get_docs().items():
        for term in doc.get_term_list():
            try:
                ntk[term] += 1
            except KeyError:
                ntk[term] = 1
    
    #calculate N and R
                    
    No_docs = coll.get_num_docs()
    R = 0
    for id, fre in ben.items():
        if ben[id] > 0:
            R += 1
    
    for id, rtk in T.items():
        T[id] = ((rtk+0.5) / (R-rtk + 0.5)) / ((ntk[id]-rtk+0.5)/(No_docs-ntk[id]-R+rtk +0.5)) 

    #calculate the mean of w4 weights.
    meanW5= 0
    for id, rtk in T.items():
        meanW5 += rtk
    meanW5 = meanW5/len(T)

    #Features selection
    Features = {t:r for t,r in T.items() if r > meanW5 + theta }
    return Features
    
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
    benFile = open('PTraining_benchmark.txt') # the pesudo relevance judgements
    file_ = benFile.readlines()

    ben={}
    for line in file_:
        line = line.strip()
        lineList = line.split()
        ben[lineList[1]]=float(lineList[2])

    benFile.close()
    theta = 3.5
    bm25_weights = w5(coll_, ben, theta)
    
    
    wFile = open('PModel_w5_R102.dat', 'w')
    for (k, v) in sorted(bm25_weights.items(), key=lambda x: x[1], reverse=True):
        wFile.write(k +' '+ str(v) +'\n')
    wFile.close()

            
