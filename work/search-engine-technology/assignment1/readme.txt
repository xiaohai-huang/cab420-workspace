##IMPORTANT##: Python3.9 or above is required to run the solution.!!!!!!!!!

Note: the file `get_outputs.sh` contains the commands to obtain the txt output files.

===============
Question 1

USAGE: python Q1.py <coll-folder-path>
    - coll-folder-path: the folder that contains the documents.

Example: `python Q1.py Rnews_t120`
===============
Question 2

USAGE: python Q2.py <query>
    - query: A query in raw text.

Note: If no query is supplied. Print out all terms (with its value of tf*idf weight in 
descending order). Otherwise, use the abstract model of ranking (Eq. (2)) to calculate a score for each document.

Examples: 
    - `python Q2.py "USA: RESEARCH ALERT - Minnesota Mining cut."`
    - `python Q2.py`
===============
Question 3

USAGE: python Q3.py <query>
    - query: the query to be tested.

Example: `python Q3.py "Deaths mining accidents"`