import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import pickle_manager
from mlxtend.frequent_patterns import association_rules

#Load data into item sets
df = pickle_manager.load_pickle("pickles\organised_ratings.pickle.lz4")

for y in range(1,11):
    for x in range(1,6):
        print("Movie Rating: ", x)
        dataSet = []
        for i in range(1,df['userId'].max()+1):
            itemSet = df.loc[(df['rating'] == x) & (df['userId'] == i)]['imdbId'].tolist()
            dataSet.insert(0,itemSet)


        #Transform the data into the correct format for the libary
        te = TransactionEncoder()
        te_ary = te.fit(dataSet).transform(dataSet)
        new_df = pd.DataFrame(te_ary, columns=te.columns_)
        #Find the frequent pattern sets and generate the assoication rules
        support = 0.01*y
        print("Support Percentage", support)
        frequent_itemsets = fpgrowth(new_df, min_support=support,use_colnames=True)
        print("Number of frequent sets:", len(frequent_itemsets))
        if(len(frequent_itemsets) > 0):
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
            print("Number of rules:", len(rules))