from itertools import islice


q_terms,  d_terms, labels = train_data
for q, d, s in islice(zip(q_terms,  d_terms, labels), 120, 140):
    print(q, d, s)
