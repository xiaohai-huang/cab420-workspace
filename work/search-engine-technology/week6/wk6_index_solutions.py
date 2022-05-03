import glob, os
import string
from stemming.porter2 import stem

def index_docs(inputpath,stop_words):
    Index = {}    # initialize the index
    os.chdir(inputpath)
    for file_ in glob.glob("*.xml"):
        start_end = False
        for line in open(file_):
            line = line.strip()
            if(start_end == False):
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1]
                            break  
                if line.startswith("<text>"):
                    start_end = True  
            elif line.startswith("</text>"):
                break
            else:
                line = line.replace("<p>", "").replace("</p>", "")
                line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
                for term in line.split():
                    term = stem(term.lower())
                    if len(term) > 2 and term not in stop_words:
                        try:
                            try:
                                Index[term][docid] += 1
                            except KeyError:
                                Index[term][docid]=1
                        except KeyError:  
                            Index[term] = {docid:1} 
    return Index


#Index1 = {'argentin': {'6146': 1}, 'bond': {'6146': 1}, 'slight': {'6146': 2}, 'higher': {'6146': 1}, 'small': {'6146': 1}, 'technic': {'6146': 2}, 'bounc': {'6146': 2}, 'wednesday': {'6146': 1}, 'amid': {'6146': 1}, 'low': {'6146': 1}, 'volum': {'6146': 1}, 'trader': {'6146': 3}, 'larg': {'6146': 1}, 'foreign': {'6146': 1}, 'bank': {'6146': 1}, 'open': {'6146': 1, '741299': 3}, 'expect': {'6146': 3}, 'price': {'6146': 1}, 'chang': {'6146': 1}, 'much': {'6146': 1}, 'dure': {'6146': 1, '741299': 1}, 'session': {'6146': 1}, 'market': {'6146': 2}, 'move': {'6146': 1}, 'news': {'6146': 1}, 'percent': {'6146': 1}, 'dollar': {'6146': 1}, 'denomin': {'6146': 1}, 'bocon': {'6146': 1}, 'prevision': {'6146': 1}, 'due': {'6146': 2}, 'rose': {'6146': 2}, 'argentina': {'6146': 2}, 'frb': {'6146': 1}, 'quot': {'6146': 2, '741299': 4}, 'general': {'6146': 1}, 'uncertainti': {'6146': 1}, 'point': {'6146': 1, '741299': 1}, 'event': {'6146': 1}, 'wait': {'6146': 1}, 'includ': {'6146': 1}, 'passag': {'6146': 1}, 'govern': {'6146': 1}, 'new': {'6146': 1}, 'econom': {'6146': 1}, 'measur': {'6146': 1}, 'through': {'6146': 1}, 'congress': {'6146': 1}, 'now': {'6146': 1}, 'until': {'6146': 1}, 'earli': {'6146': 1}, 'octob': {'6146': 1}, 'addit': {'6146': 1}, 'await': {'6146': 1}, 'meet': {'6146': 1}, 'friday': {'6146': 1}, 'between': {'6146': 1}, 'economi': {'6146': 1}, 'minist': {'6146': 1}, 'roqu': {'6146': 1}, 'fernandez': {'6146': 1}, 'intern': {'6146': 1}, 'monetari': {'6146': 1}, 'fund': {'6146': 1}, 'deleg': {'6146': 1}, 'fiscal': {'6146': 1}, 'deficit': {'6146': 1}, 'axel': {'6146': 1}, 'bugg': {'6146': 1}, 'bueno': {'6146': 1}, 'air': {'6146': 1}, 'newsroom': {'6146': 1}, 'lehto': {'741299': 2}, 'finland': {'741299': 1}, 'steve': {'741299': 1}, 'soper': {'741299': 2}, 'britain': {'741299': 1}, 'drove': {'741299': 1}, 'ail': {'741299': 1}, 'mclaren': {'741299': 1}, 'victori': {'741299': 2}, 'fifth': {'741299': 1}, 'round': {'741299': 1}, 'world': {'741299': 1}, 'championship': {'741299': 1}, 'sunday': {'741299': 1}, 'beat': {'741299': 1}, 'merced': {'741299': 1}, 'german': {'741299': 2}, 'bernd': {'741299': 1}, 'schneider': {'741299': 2}, 'austrian': {'741299': 1}, 'alexand': {'741299': 1}, 'wurz': {'741299': 1}, 'second': {'741299': 2}, 'enabl': {'741299': 1}, 'lead': {'741299': 3}, 'overal': {'741299': 1}, 'stand': {'741299': 1}, 'over': {'741299': 2}, 'mount': {'741299': 1}, 'strong': {'741299': 1}, 'challeng': {'741299': 1}, 'struggl': {'741299': 2}, 'leader': {'741299': 1}, 'final': {'741299': 1}, 'minut': {'741299': 1}, 'four': {'741299': 1}, 'hour': {'741299': 1}, 'race': {'741299': 2}, 'car': {'741299': 3}, 'handl': {'741299': 1}, 'caus': {'741299': 1}, 'broken': {'741299': 1}, 'undertray': {'741299': 1}, 'manag': {'741299': 1}, 'hold': {'741299': 1}, 'win': {'741299': 1}, 'mid': {'741299': 1}, 'downpour': {'741299': 1}, 'ardenn': {'741299': 1}, 'mountain': {'741299': 1}, 'thought': {'741299': 1}, 'everyon': {'741299': 1}, 'els': {'741299': 1}, 'drive': {'741299': 1}, 'dri': {'741299': 1}, 'weather': {'741299': 1}, 'tyre': {'741299': 2}, 'joke': {'741299': 1}, 'afterward': {'741299': 1}, 'swap': {'741299': 1}, 'rain': {'741299': 1}, 'exact': {'741299': 1}, 'right': {'741299': 1}, 'time': {'741299': 1}, 'abl': {'741299': 1}, 'push': {'741299': 1}, 'hard': {'741299': 1}, 'big': {'741299': 1}, 'third': {'741299': 1}, 'finish': {'741299': 1}, 'porsch': {'741299': 1}, 'franc': {'741299': 1}, 'bob': {'741299': 1}, 'wollek': {'741299': 1}, 'yannick': {'741299': 1}, 'dalma': {'741299': 1}, 'belgian': {'741299': 2}, 'thierri': {'741299': 1}, 'boutsen': {'741299': 1}, 'former': {'741299': 1}, 'formula': {'741299': 1}, 'one': {'741299': 1}, 'driver': {'741299': 1}, 'switch': {'741299': 1}, 'normal': {'741299': 1}, 'share': {'741299': 1}, 'han': {'741299': 1}, 'stuck': {'741299': 1}, 'follow': {'741299': 1}, 'power': {'741299': 1}, 'steer': {'741299': 1}, 'failur': {'741299': 1}}
#print(Index1)


def doc_at_a_time(I, Q):  # index I is a Dirctionary of term:Directionary of (itemId:freq)
    L={}    # L is the selected inverted list
    R={}    # R is a directionary of docId:relevance
    for list in I.items():
        for id in list[1].items(): # get all document IDs with value 0
            R[id[0]]=0
        if (list[0] in Q):     # select inverted lists based on the query
                L[list[0]]= I[list[0]]
    for (d, sd) in R.items():
        for (term, f) in L.items():
            if (d in f):
                sd = sd + f[d]*Q[term]
        R[d] = sd
    return R


def term_at_a_time(I, Q):  # index I is a Dirctionary of term:Directionary of (itemId:freq)
    L={}    # L is the selected inverted list
    R={}    # R is a directionary of docId:relevance
    for list in I.items():
        for id in list[1].items(): # get all document IDs with value 0
            R[id[0]]=0
        if (list[0] in Q):     # select inverted lists based on the query
                L[list[0]]= I[list[0]]
    for (term, li) in L.items():  # traversal of the selected inverted list
        for (d, f) in li.items(): # for each occurence of doc, update R 
                R[d] = R[d]  + f*Q[term]
    return R



if __name__ == '__main__':

    import sys

    if len(sys.argv) != 2:
        sys.stderr.write("USAGE: %s <coll-file>\n" % sys.argv[0])
        sys.exit()
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    Index = index_docs(sys.argv[1], stop_words) #create an index for all terms in <text>, data structure {'w1':{'ID1':2, 'ID2':1}, 'w2':{'ID3':1, 'ID1':3}}
    """    for term in coll.items():
        print "Term --- %s" % (term[0])
        for id in coll[term[0]].items(): 
            print "   Document ID: %s and frequency: %d" % (id[0], id[1]) """
    #Query = {'leaderboard':1, 'british':1}
    #print(Index)
    Query = {'formula':1, 'one':1} 
    result1 = doc_at_a_time(Index, Query)
    result2 = term_at_a_time(Index, Query)
    x1 = sorted(result1.items(), key=lambda x: x[1],reverse=True)
    x2 = sorted(result2.items(), key=lambda x: x[1],reverse=True)
    print('Document_at_a_time result------')
    for (id, w) in x1:
        if w>0:
            print('Document ID: '+id + ' and relevance weight: ' + str(w))
    print('Term_at_a_time result ------')
    for (id, w) in x2:
        if w>0:
            print('Document ID: ' + id + ' and relevance weight: ' + str(w))

