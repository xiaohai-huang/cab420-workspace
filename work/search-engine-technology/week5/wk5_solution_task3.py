#week 5 workshop solution for task 3

#define a function to calculate df
def c_df(docs):
    """Calculate DF of each term in vocab and return as term:df dictionary."""
    df_ = {}
    for id, doc in docs.items():
        for term in doc.keys():
            try:
                df_[term] += 1
            except KeyError:
                df_[term] = 1
    return df_

# the document collection is represented as a dict
docs = {'D1':{'term1':3, 'term4':5, 'term5':7},'D2':{'term1':5, 'term2':3, 'term3':4, 'term4':6}, 'D3':{'term3':5, 'term4':4, 'term5':6}, 'D4':{'term1':9, 'term4':1, 'term5':2}, 'D5':{'term2':1, 'term4':3, 'term5':2},'D6':{'term1':3, 'term3':2, 'term4':4, 'term5':4}}
# test the input
print(docs)
print('\n')
# call the function
x = c_df(docs)
print('Display terms df from term 1 to term 5: \n')
y = {k: v for k, v in sorted(x.items(), key=lambda item: item[0])}
print(y)
