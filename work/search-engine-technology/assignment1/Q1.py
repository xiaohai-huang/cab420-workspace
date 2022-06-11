from glob import glob
import string
from stemming.porter2 import stem

class BowDoc:
    """Bag of Words representation of a document."""
    def __init__(self, docID:str="", terms:dict[str, int]=None, doc_len:int=0):
        """Initialize a BowDoc object.

        @docId: A docID variable which is simply assigned by the value of 'itemid' in <newsitem …>.
        @terms: key-value pair of (String term: int frequency)
        @doc_len: The total number of words in a document.
        """
        self.docID = docID
        self.terms = terms
        self.doc_len = doc_len
    
    def getDocID(self):
        """Getter for docID"""
        return self.docID

    def getTerms(self):
        """Getter for terms"""
        return self.terms

    def setDocLen(self, doc_len):
        """Setter for doc_len"""
        self.doc_len = doc_len

    def getDocLen(self):
        """Getter for doc_len"""
        return self.doc_len
    
    def addTerm(self, new_term:str):
        """add new term or increase term frequency when the term occur again."""
        if new_term in self.terms:
            self.terms[new_term] += 1
        else:
            self.terms[new_term] = 1

def process_line(line:str, on_term_process):
    """Process a line of string.
    
    @line: A line of words.
    @on_term_process: A function that will be call back when processing a term.
    """
    # remove \n and spaces
    line = line.strip()

    # exclude p tags
    line = line.replace("<p>","").replace("</p>", "")

    # discard punctuations and/or numbers
    line = line.translate(str.maketrans("", "", string.digits))
    line = line.translate(str.maketrans(string.punctuation, " "* len(string.punctuation)))

    for term in line.split():
        term = stem(term.lower())
        on_term_process(term)

def parse_rcv_coll(input_path:str, stop_words:list[str]) -> dict[str, BowDoc]:
    """Parse a data collection in the given folder and 
    build up a collection of BowDoc objects for the given dataset

    @input_path: The folder that contains the dataset in XML format.
    @stop_words: A list of common English words.

    @return: A collection of BowDoc objects. A dictionary structure with docID as key and BowDoc object as value. 
    """
    # build up a collection of BowDoc objects for the given dataset
    # docID as key and BowDoc object as value
    BowDoc_collection:dict[str, BowDoc] = {} 
    for file_path in glob(f"{input_path}/*.xml"):
        with open(file_path, "r", encoding="UTF-8") as file:
            word_count = 0
            docID = ""

            def process_term(term:str):
                nonlocal word_count
                word_count += 1
                # Stopping words removal and add to BowDoc
                if len(term) > 2 and (term not in stop_words):
                    BowDoc_collection[docID].addTerm(term)

            text_tag_start = False
            for line in file:
                # remove \n and spaces
                line = line.strip()

                # obtain docID from itemid in <newsitem>
                if not docID:
                    if line.startswith("<newsitem "):
                        for part in line.split():
                            if part.startswith("itemid="):
                                docID = part.split("=")[1].strip("\"")
                                BowDoc_collection[docID] = BowDoc(docID,{})

                 # look for the content of <text></text>
                if line.startswith("<text>"):
                    text_tag_start = True
                    continue
                elif line.startswith("</text>"):
                    text_tag_start = False
                    BowDoc_collection[docID].setDocLen(word_count)
                    break
                
                # tokenize <text></text>
                if text_tag_start:
                    process_line(line, process_term)
                   
    return BowDoc_collection

def parse_query(query0:str, stop_words:list[str]) -> dict[str, int]:
    """Parse a query and return a term frequency dictionary. {term, num_occurrences}
    
    @query0: A simple sentence or title.
    @stop_words: A list of stop words.

    @return: A term frequency dictionary. {term:frequency}
    """
    d = BowDoc(terms={})
    def process_term(term:str):
        # Stopping words removal and add to BowDoc
        if len(term) > 2 and (term not in stop_words):
            d.addTerm(term)
        
    process_line(query0, process_term)
    return d.getTerms()

def main():
    """A main function to test function parse_rcv_coll()"""
    import sys
    if len(sys.argv) != 2:
        sys.stderr.write("USAGE: python %s <coll-folder-path>\n" % sys.argv[0])
        sys.exit()
    coll_folder_path = sys.argv[1]

    stopwords_f = open('common-english-words.txt', 'r', encoding="UTF-8")
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()

    # docs = parse_rcv_coll("Rnews_t120", stop_words)
    docs = parse_rcv_coll(coll_folder_path, stop_words)

    for docID, doc in sorted(docs.items()):
        print(f"Document {docID} contains {len(doc.getTerms())} indexing terms and have total {doc.getDocLen()} words")
        # sorts index terms (by frequency)
        sorted_terms = sorted(doc.getTerms().items(),reverse=True, key=lambda item: item[1])
        # prints out a term:freq list
        for term, count in sorted_terms:
            print(f"{term} : {count}")


if __name__ == "__main__":
    main()
