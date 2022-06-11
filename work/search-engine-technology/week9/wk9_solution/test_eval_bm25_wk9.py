# Solutions for Task 2 and Task 3.
# Task 2 uses Test_set and features selected by Task 1 (file Model_w5_R102.dat).
#   It generates a ranking for documents in Test_set.
# Task 3 uses the ranking provided by the relevance model (Task 1 & Task 2), and
#    the relevance judgement Test_benchmark.txt to calculate Recall and Precision at rank positions and average precision.


import math
from stemming.porter2 import stem


    
def BM25Testing(coll, features):
    ranks={}
    for id, doc in coll.get_docs().items():
        Rank = 0
        for term in features.keys():
            if term in doc.get_term_list():
                try:
                    ranks[id] += features[term]
                except KeyError:
                    ranks[id] = features[term]
    return ranks
    
if __name__ == "__main__":

    import sys
    import os
    import coll
    #import df

    if len(sys.argv) != 2:
        sys.stderr.write("USAGE: %s <coll-file>\n" % sys.argv[0])
        sys.exit()
    coll_fname = sys.argv[1]

    #pre-processing documents
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    coll_ = coll.parse_rcv_coll(coll_fname, stop_words)
    

    # get features
    os.chdir('..')
    featureFile = open('Model_w5_R102.dat')
    file_ = featureFile.readlines()
    features={}
    for line in file_:
        line = line.strip()
        lineList = line.split()
        features[lineList[0]]=float(lineList[1])
    featureFile.close()

    # obtain ranks for all documents 
    ranks = BM25Testing(coll_, features)
    
    wFile = open('R102_test_ranks.dat', 'w')
    for (d, v) in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
        wFile.write(d +' '+ str(v) +'\n')
    wFile.close()



    # task 3 evaluation
    # get the benchmark
    benFile = open('Test_benchmark.txt')
    #benFile = open('Training_benchmark.txt')
    file_ = benFile.readlines()
    ben={}
    for line in file_:
        line = line.strip()
        lineList = line.split()
        ben[lineList[1]]=float(lineList[2])
    benFile.close()
    #print(ben)
    
    # number documents 
    rank1={}
    i=1
    for line in open("R102_test_ranks.dat"):
        line = line.strip()
        line1 = line.split()
        rank1[str(i)] = line1[0]
        i = i + 1

    #print(rank1)
    
    print("For task 3:")
    ri = 0
    map1 = 0.0
    R = len([id for (id,v) in ben.items() if v>0])
    for (n,id) in sorted(rank1.items(), key=lambda x: int(x[0])):
        if (ben[id]>0):
            ri =ri+1
            pi = float(ri)/float(int(n))
            recall = float(ri)/float(R)
            map1 = map1 + pi
            print("At position " + str(int(n)) + ", precision= " + str(pi) + ", recall= " + str(recall))
    map1 = map1/float(ri)
    print("---The average precision = " + str(map1))
