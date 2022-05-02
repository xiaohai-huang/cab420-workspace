import glob, os
import string


def parse_coll(inputpath, stop_ws):
    coll = {}
    os.chdir(inputpath)
    myfile = open('6146.xml')
    curr_doc = {}
    start_end = False
    file_ = myfile.readlines()
    word_count = 0  # wk3
    for line in file_:
        line = line.strip()
        # print(line)
        if (start_end == False):
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
            line = line.translate(str.maketrans('', '', string.digits)).translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            line = line.replace("\\s+", " ")
            for term in line.split():
                word_count += 1  # wk3
                term = term.lower()  # wk3
                if len(term) > 2 and term not in stop_words:  # wk3
                    try:
                        curr_doc[term] += 1
                    except KeyError:
                        curr_doc[term] = 1
    myfile.close()
    return (word_count, {docid: curr_doc})
    # return a tuple, the first element is the number of words in <text> and
    # the second one is a dictionary that includes only one pair of doc_id and a dictionary of term_frequency pairs


if __name__ == '__main__':

    import sys

    if len(sys.argv) != 2:
        sys.stderr.write("USAGE: %s <coll-file>\n" % sys.argv[0])
        sys.exit()

    stopwords_f = open('common-english-words.txt', 'r')  # wk3
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()

    x = parse_coll(sys.argv[1], stop_words)

    for doc in x[1].items():
        print('Document itemid: ' + doc[0] + ' contains: ' + str(x[0]) + ' words and ' + str(len(doc[1])) + ' terms')
    # show terms (in ascending order) and their frequency, the number of occurances in <text>
    # please note it is not possible to sort a dictionary, only to get a representation of a dictionary that is sorted.

    print('------Terms and Their Frequencies-----')
    for doc in x[1].items():
        doc1 = {k: v for k, v in sorted(doc[1].items(), reverse=False)}
        for term, freq in doc1.items():
            print(term + ' : ' + str(freq))

