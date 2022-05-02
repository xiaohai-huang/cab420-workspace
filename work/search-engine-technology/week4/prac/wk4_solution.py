import glob, os
import string
from stemming.porter2 import stem  ## for wk 4

#Node class of document
class Doc_Node:
    def __init__(self, data, next=None):
        self.data=data
        self.next=next

#Linked List calss   
class List_Docs:
    def __init__(self, hnode):
        self.head=hnode

    def insert(self, nnode):
        if self.head != None:
            p = self.head
            while p.next != None:
                p=p.next
            p.next=nnode
            
    def lprint(self):
        if self.head != None:
            p = self.head
            while p!= None:
                print('(ID-'+p.data[0] + ':',end =" " )
                print(str(len(p.data[1]))+' terms)',end =" " )
                if p.next != None:
                    print ('-->', end=" ")
                p=p.next
        else:
            print('The list is empty!')



def parse_coll(fn, stop_ws):
    coll = {}    
    #os.chdir(inputpath)
    #myfile=open('741299newsML.xml')
    myfile=open(fn)
    #myfile=open('C:\\python27\\py_CAB431_201\\data\\741299newsML.xml', 'r')
    curr_doc = {}
    start_end = False
    #docid='741299'
    file_=myfile.readlines()
    #word_count = 0 #wk3
    for line in file_:
        line = line.strip()
        #print(line)
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
            line = ' '.join(line.split())
            for term in line.split():
                #word_count += 1 #wk3
                term = stem(term.lower()) ## for wk 4
                #term = term.lower() #wk3
                if len(term) > 2 and term not in stop_words: #wk3
                    try:
                        curr_doc[term] += 1
                    except KeyError:
                        curr_doc[term] = 1
    myfile.close()
    dn=Doc_Node((docid,curr_doc), None)
    return(dn)
    # return a tuple, the first element is the number of words in <text> and
    # the second one is a dirctionary that includes only one pair of doc_id and a disctionary of term_frequency pairs 


if __name__ == '__main__':

    import sys
    if len(sys.argv) != 2:
        sys.stderr.write("USAGE: %s <coll-file>\n" % sys.argv[0])
        sys.exit()

    stopwords_f = open('common-english-words.txt', 'r') # wk3
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    os.chdir(sys.argv[1])
    fn1='6146.xml'
    fn2='741299newsML.xml'
    xn1 = parse_coll(fn1,stop_words)
    xn2 = parse_coll(fn2,stop_words)
    ll= List_Docs(xn1)
    ll.insert(xn2)
    ll.lprint()



    

        
