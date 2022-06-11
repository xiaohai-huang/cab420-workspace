import glob, os
import string

# this function returns relevant documents A and retrieved documents B
def get_re_docs(inputpath):
    coll = {}    
    #os.chdir(inputpath)
    A = {}
    B= {}
    R1 = {}
    R2 = {}
    for line in open("Training_benchmark.txt"):
        line = line.strip()
        line1 = line.split()
        A[line1[1]] = int(float(line1[2]))
    for line in open("PTraining_benchmark.txt"):
        line = line.strip()
        line1 = line.split()
        B[line1[1]] = int(float(line1[2]))
    return (A,B)



if __name__ == '__main__':

    import sys
    # please note I have not created a folder for the input files.
    # So the system argv doe not make sense. When test it, any folder name is ok.
    if len(sys.argv) != 2:
        sys.stderr.write("USAGE: %s <coll-file>\n" % sys.argv[0])
        sys.exit()

    (rel_doc, retrived_doc) = get_re_docs(sys.argv[1])
    R = 0
    for (x,y) in rel_doc.items():
        if (y==1):
            R= R+1
    print("the number of relevant docs: " + str(R))
        
    R1 = 0
    for (x,y) in retrived_doc.items():
        if (y==1):
            R1= R1+1
    print("the number of retrieved docs: "+ str(R1))

    RR1 = 0
    for (x,y) in rel_doc.items():
        if (y==1) & (retrived_doc[x]==1):
            RR1= RR1+1
    print("the number of retrieved docs that are relevant: " + str(RR1))
    r = float(RR1)/float(R)
    p = float(RR1)/float(R1)
    F1 = 2*p*r/(p+r)
    print("recall = " + str(r))
    print("precision = " + str(p))
    print("F-Measure = " + str(F1))
    
