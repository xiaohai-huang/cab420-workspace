import glob, os
from pydoc import doc
import string


def task1(inputpath):
    coll = {}  
    # print(os.getcwd())
    import pathlib
    # print(pathlib.Path(__file__).parent.resolve()+inputpath)
    os.chdir(str(pathlib.Path(__file__).parent.resolve())+"/"+inputpath)
    # for task 2
    A = {}
    B= {}
    # for task 3
    C = {}
    R = {}
    
    # for task 2
    for line in open("relevance_judgments.txt"):
        line = line.strip()
        line1 = line.split()
        A[line1[1]] = int(float(line1[2]))
    for line in open("binary_output.txt"):
        line = line.strip()
        line1 = line.split()
        B[line1[1]] = int(float(line1[2]))
    # for task 3
    for line in open("ranked_output.txt"):
        line = line.strip()
        line1 = line.split()
        C[line1[1]] = float(line1[2])
    # get the top-10 document in terms of {rankingNO: documentID, ...}
    i=1
    for (k,v) in sorted(C.items(), key=lambda x:x[1], reverse=True):
        R[i] = k
        i = i+1
        if i>10:
            break
    #print(sorted(C.items(), key=lambda x: float(x[1])))
    print(R)
    return (A,B,R)

def task3(rel_doc, retrieved_doc, ranked_doc):
    # a set of relevant doc
    A = set()
    for docID, relevance_judgement in rel_doc.items():
        if relevance_judgement == 1:
            A.add(docID)
    # a set of retrieved doc
    B = set()
    for docID, relevance_value in retrieved_doc.items():
        if relevance_value == 1:
            B.add(docID)

    avg_precision = 0
    count = 0
    for n, docID in ranked_doc.items():
        # only calculate these measures if a relevant document was retrieved
        if docID in A:

            # B_n is the set of top-n documents in the output of the IR model
            B_n = set(list(ranked_doc.values())[:n])
            A_and_B_n = len(set.intersection(A, B_n))
            recall = A_and_B_n / len(A)
            precision = A_and_B_n / n
            print("At position " + str(int(n)) + " docID: " + docID + ", precision= " + str(precision) +f", recall={recall}")
            avg_precision += precision
            count += 1
    print(f"avg precision={avg_precision/count}")
if __name__ == '__main__':

    import sys
    # if len(sys.argv) != 2:
    #     sys.stderr.write("USAGE: %s <coll-file>\n" % sys.argv[0])
    #     sys.exit()
    (rel_doc, retrived_doc, ranked_doc) = task1("rel_data")
    task3(rel_doc, retrived_doc, ranked_doc)
    raise Exception
    # for task 2
    R = 0
    for (x,y) in rel_doc.items():
        if (y==1):
            R= R+1
    print("The number of relevant documents: " + str(R))
        
    R1 = 0
    for (x,y) in retrived_doc.items():
        if (y==1):
            R1= R1+1
    print("The number of retrieved documents: "+ str(R1))

    RR1 = 0
    for (x,y) in retrived_doc.items():
        if (y==1) & (rel_doc[x]==1):
            RR1= RR1+1
  
    print("The number of retrieved documents that are relevant: " + str(RR1))
    r = float(RR1)/float(R)
    p = float(RR1)/float(R1)
    F1 = 2*p*r/(p+r)
    print("recall = " + str(r))
    print("precision = " + str(p))
    print("F-Measure = " + str(F1))

    # code for task 3
    print("For task 3:")
    ri = 0
    ap1 = 0.0
    ranked_doc = sorted(ranked_doc.items(), key=lambda x: int(x[0]))
    for (n,id) in ranked_doc:
        if (rel_doc[id]==1):
            ri =ri+1
            pi = float(ri)/float(int(n))
            ap1 = ap1 + pi
            print("At position " + str(int(n)) + "docID: " + id + ", precision= " + str(pi))
    ap1 = ap1/float(ri)
    print("The average precision = " + str(ap1))
    
