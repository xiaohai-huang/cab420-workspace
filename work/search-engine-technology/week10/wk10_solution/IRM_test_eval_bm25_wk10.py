# Task 2 - test the IR model's ranking results

import math
from stemming.porter2 import stem

if __name__ == "__main__":

    import sys
    import os
    #import coll
    #import df


    # task 2 evaluation
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
    for line in open("IRModel_R102.dat"):
        line = line.strip()
        line1 = line.split()
        rank1[str(i)] = line1[0]
        i = i + 1

    #print(rank1)
    
    print("For task 2:")
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
