# Main function for Task 2

if __name__ == "__main__":

    import sys
    import os
    import coll
    import df

    writeFile = open('PTraining_benchmark.txt', 'a')
    bm25_threshold = 1.0
    # you may change it to a different value, e.g., 5.0 for BaselineModel_R102_v2.dat

    datFile = open('BaselineModel_R102.dat')
    file_=datFile.readlines()
    for line in file_:
        line = line.strip()
        lineStr = line.split()
        if float(lineStr[1]) > bm25_threshold:
            writeFile.write('R102 ' + lineStr[0] + ' 1' +'\n')
        else:
            writeFile.write('R102 ' + lineStr[0] + ' 0' +'\n')
    writeFile.close()
    datFile.close()
  
