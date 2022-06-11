from math import sqrt, log
from Q1 import BowDoc, parse_query, parse_rcv_coll

def calc_df(coll:dict[str, BowDoc]):
    """calculate document-frequency (df) for a given BowDoc collection coll and return a {term:df, …} dictionary. 
    @coll: A BowDoc collection. {docID:BowDoc}

    @return: A {term:df, …} dictionary.
    """
    # {term: count}
    df = {} 
    output = ""
    
    for doc in coll.values():
        terms = doc.getTerms()
        for term in terms.keys():
            if term in df:
                df[term] += 1
            else:
                df[term] = 1
    
    # sort them in descending order
    
    df = { k:v for k,v in sorted(df.items(), key=lambda item: item[1], reverse=True) }

    output += f"There are {len(coll)} documents in this data set and contains {len(df)} terms\n"
    for term, count in df.items():
        output += f"{term} : {count}\n"
    # print(output)
    return df

def tfidf(doc:BowDoc, df:dict[str, int], ndocs:int) -> dict[str, float]:
    """calculate tf*idf value (weight)
    
    @doc: A BowDoc instance.
    @df: A {term: df} dictionary.
    @ndocs: The number of documents in a given BowDoc collection.

    @ returns a {term:tfidf_weight , …} dictionary for the given document doc.
    """
    tf_idf = {}
    denominator = 0
    terms = doc.getTerms()
    for term, count in terms.items():
        denominator += ((log(count) + 1) * log(ndocs / df[term]))**2
    
    denominator = sqrt(denominator)
    for term, tf in doc.getTerms().items():
        numerator = (log(tf) + 1) * log(ndocs/df[term])
        tf_idf[term] = numerator / denominator

    return tf_idf



def main():
    """A main function to print out all terms (with its value of tf*idf weight in 
    descending order) for each document in Rnews_t120"""
    import sys
    stopwords_f = open('common-english-words.txt', 'r', encoding="UTF-8")
    stop_words = stopwords_f.read().split(',')
    docs = parse_rcv_coll("Rnews_t120", stop_words)
    
    # Number of documents
    N = len(docs)
    df = calc_df(docs)

    if len(sys.argv) == 1:
        for doc in docs.values():
            terms = doc.getTerms()
            tf_idf = tfidf(doc, df, N)

            print(f"Document {doc.getDocID()} contains {len(terms)} terms")
            for term, weight in sorted(tf_idf.items(), key=lambda item: item[1], reverse=True):
                print(f"{term} : {weight}")
    elif len(sys.argv) == 2:
        # queries = ["USA: RESEARCH ALERT - Minnesota Mining cut.", 
        #            "SOUTH AFRICA: Death toll reaches 24 in S.African mine clashes.",
        #            "SOUTH AFRICA: Three killed in new clashes at S.Africa gold mine."]
        query = sys.argv[1]
        # {docID:relevance}
        Q = parse_query(query, stop_words)
        R = {docID:0 for docID in docs}
        for doc in docs.values():
            terms = doc.getTerms()
            tf_idf = tfidf(doc, df, N)
            for term in Q:
                if term in tf_idf:
                    R[doc.getDocID()] += tf_idf[term] * Q[term]            
        
        print(f"The Ranking Result for query: {query}\n")
        for k,v in sorted(R.items(), key=lambda item:item[1] ,reverse=True):
            print(f"{k} : {v}")
            
        
if __name__ == "__main__":
    main()
    