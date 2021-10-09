import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from time import perf_counter
import pickle_manager
from mlxtend.frequent_patterns import association_rules

#Load data into item sets
df = pickle_manager.load_pickle("pickles\organised_ratings.pickle.lz4")


for x in range(6,6):
    print("Movie Rating: ", x)
    dataSet = []
    for i in range(1,df['userId'].max()+1):
        itemSet = df.loc[(df['rating'] == x) & (df['userId'] == i)]['imdbId'].tolist()
        dataSet.insert(0,itemSet)
    print(dataSet)

    #Transform the data into the correct format for the libary
    te = TransactionEncoder()
    te_ary = te.fit(dataSet).transform(dataSet)
    new_df = pd.DataFrame(te_ary, columns=te.columns_)

    #Find the frequent pattern sets and generate the assoication rules
    start = perf_counter()
    frequent_itemsets = fpgrowth(new_df, min_support=0.05,use_colnames=True)
    #print(frequent_itemsets.sort_values('support',ascending=False))
    print("Number of frequent sets:", len(frequent_itemsets))
    if(len(frequent_itemsets) > 0):
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
        print("Number of rules:", len(rules))
        #print(rules)
    end = perf_counter()
    print(end - start, start, end)


results_df = pd.DataFrame(data={'Datset Size':[], 'Time Taken':[]})
for y in range(20,123):
    for x in range(0,6):
        #print("Movie Rating: ", x)
        dataSet = []
        for i in range(1,df['userId'].max()+1):
            itemSet = df.loc[(df['rating'] == x) & (df['userId'] == i)]['imdbId'].tolist()
            dataSet.insert(0,itemSet)
        dataSet = dataSet[0:y*5]
        print("Dataset size: ", len(dataSet))

        #Transform the data into the correct format for the libary
        te = TransactionEncoder()
        te_ary = te.fit(dataSet).transform(dataSet)
        new_df = pd.DataFrame(te_ary, columns=te.columns_)

        #Find the frequent pattern sets and generate the assoication rules
        start = perf_counter()
        frequent_itemsets = fpgrowth(new_df, min_support=0.03,use_colnames=True)
        #print(frequent_itemsets.sort_values('support',ascending=False))
        print("Number of frequent sets:", len(frequent_itemsets))
        #if(len(frequent_itemsets) > 0):
        #    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
        #   print("Number of rules:", len(rules))
            #print(rules)
        end = perf_counter()
        #print(end - start, start, end)
        newRow = {'Datset Size': len(dataSet), 'Time Taken': end-start}
        results_df = results_df.append(newRow, ignore_index=True)


results_df.to_csv('Library.csv', index=False, )