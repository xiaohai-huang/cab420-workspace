# Task 2 Solution -  a BM25 based IR model.

import math
from stemming.porter2 import stem

def avg_doc_len(coll):
    tot_dl = 0
    for id, doc in coll.get_docs().items():
        tot_dl = tot_dl + doc.get_doc_len()
    return tot_dl/coll.get_num_docs()

# Task 2 - BM25 based IR model

def bm25(coll, q, df):
    bm25s = {}
    avg_dl = avg_doc_len(coll)
    no_docs = coll.get_num_docs()
    for id, doc in coll.get_docs().items():
        query_terms = q.split()
        qfs = {}        
        for t in query_terms:
            term = stem(t.lower())
            try:
                qfs[term] += 1
            except KeyError:
                qfs[term] = 1
        k = 1.2 * ((1 - 0.75) + 0.75 * (doc.get_doc_len() / float(avg_dl)))
        bm25_ = 0.0;
        for qt in qfs.keys():
            n = 0
            if qt in df.keys():
                n = df[qt]
                f = doc.get_term_count(qt);
                qf = qfs[qt]
                bm = math.log(1.0 / ((n + 0.5) / (no_docs - n + 0.5)), 2) * (((1.2 + 1) * f) / (k + f)) * ( ((100 + 1) * qf) / float(100 + qf))
                # bm valuse may be negative if no_docs < 2n+1, so we may use 3*no_docs to solve this problem.
                bm25_ += bm
        bm25s[doc.get_docid()] = bm25_
    return bm25s
    
if __name__ == "__main__":

    import sys
    import os
    import coll
    import df

    if len(sys.argv) != 2:
        sys.stderr.write("USAGE: %s <coll-file>\n" % sys.argv[0])
        sys.exit()
    coll_fname = sys.argv[1] # uses Test_set directly
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    # document parsing 
    coll_ = coll.parse_rcv_coll(coll_fname, stop_words)    
    # calculate document frequency
    df_ = df.calc_df(coll_)
    # call BM25 IR model - Task 2
    bm25_1 = bm25(coll_, "Convicts repeat offenders", df_)
    print('For query Q = ' + "\"Convicts repeat offenders\"")
    os.chdir('..')
    wFile = open('IRModel_R102.dat', 'a')
    for (k, v) in sorted(bm25_1.items(), key=lambda x: x[1], reverse=True):
        wFile.write(k +' '+ str(v) +'\n')
    wFile.close()

    
    
        
