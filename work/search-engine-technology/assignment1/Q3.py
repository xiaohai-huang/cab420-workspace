from math import log
from Q1 import BowDoc, parse_query, parse_rcv_coll
from Q2 import calc_df

def avg_doc_len(coll:dict[str, BowDoc]) -> float:
    """calculate  and  return  the  average document length of all documents in the collection coll.
    
    @coll: A collection of BowDoc objects. {docID:BowDoc}.

    @return: The average document length.
    """
    total_len = 0
    N = len(coll)
    for doc in coll.values():
        total_len += doc.getDocLen()
    return total_len / N

def bm25(coll:dict[str, BowDoc], q:str, df:dict[str, int]) -> dict[str, float]:
    """Calculate  documents' BM25 score for a given original query q
    
    @coll: A collection of documents. {docID: BowDoc}.
    @q: The original query.
    @df: document frequency. {term:df}.

    @return: A dictionary of {docID:bm25_score} for all documents in collection coll.
    """
    stopwords_f = open('common-english-words.txt', 'r', encoding="UTF-8")
    stop_words = stopwords_f.read().split(',')
    scores = {}
    k_1 = 1.2
    k_2 = 100
    b = 0.75
    dl = 0
    N = 2*len(coll)
    # the average document length of a doc in the collection
    avdl = avg_doc_len(coll)
    query_term_frequency = parse_query(q, stop_words)

    for doc in coll.values():
        score = 0
        dl = doc.getDocLen()
        # frequency of terms in the document
        f = doc.getTerms()
        K = k_1 * ( (1 - b) + b *  dl/avdl)
        for term in query_term_frequency:
            f_i = 0
            n_i = 0
            if term in f:
                f_i = f[term]
            if term in df:
                n_i = df[term]
            qf_i = query_term_frequency[term]
            first_term = 1/( (n_i + 0.5)/(N - n_i + 0.5) )
            second_term = ((k_1 + 1)*f_i) / (K + f_i)
            third_term = ((k_2 + 1)*qf_i)/(k_2 + qf_i)

            score += log(first_term, 2) * second_term * third_term
        scores[doc.getDocID()] = score
    
    return scores
    
def main():
    """a main function to implement a BM25-based IR model to rank documents in
    the given document collection"""
    import sys
    if len(sys.argv) != 2:
        sys.stderr.write("USAGE: python %s <query>\n" % sys.argv[0])
        sys.exit()
    q = sys.argv[1]

    # queries = ["Deaths mining accidents", "Mentioning deaths in mining accidents", "Statistics on number of mining deaths", "ethnic clashes and resultant deaths of mine workers near a mine"]
    stopwords_f = open('common-english-words.txt', 'r', encoding="UTF-8")
    stop_words = stopwords_f.read().split(',')
    docs = parse_rcv_coll("Rnews_t120", stop_words)
    
    avdl = avg_doc_len(docs)
    print(f"Average document length for this collection is: {avdl}")
    df = calc_df(docs)

    print(f"The query is: {q}")
    bm25_scores = bm25(docs, q, df)
    print("The following are the BM25 score for each document:")
    for docID, score in sorted(bm25_scores.items(), key=lambda item:item[1], reverse=True):
        doc_len = docs[docID].getDocLen()
        print(f"Document ID: {docID}, Doc Length: {doc_len} -- BM25 Score: {score}")
    
    print("\nThe following are possibly relevant documents retrieved -")
    for docID, score in sorted(bm25_scores.items(), key=lambda item:item[1], reverse=True)[:5+1]:
        print(f"{docID} {score}")

if __name__ == "__main__":
    main()