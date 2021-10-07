import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from time import perf_counter
import pickle_manager


df = pickle_manager.load_pickle("pickles\organised_ratings.pickle.lz4")


dataSet = []
for i in range(1,df['userId'].max()+1):
    itemSet = []
    itemSet = df.loc[(df['rating'] == 5) & (df['userId'] == i)]['imdbId'].tolist()
    dataSet.insert(0,itemSet)


te = TransactionEncoder()
te_ary = te.fit(dataSet).transform(dataSet)
df = pd.DataFrame(te_ary, columns=te.columns_)

start = perf_counter()
results = fpgrowth(df, min_support=0.05,use_colnames=True)
print(results.sort_values('support',ascending=False))
print(len(results))
end = perf_counter()
print(end - start, start, end)