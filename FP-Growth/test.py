import pickle_manager
import pandas as pd
import os

df = pickle_manager.load_pickle("pickles\organised_ratings.pickle.lz4")


itemSets = []
for i in range(1,df['userId'].max()+1):
    itemSet = df.loc[(df['rating'] == 5) & (df['userId'] == i)]['imdbId'].tolist()
    itemSets.insert(0,itemSet)

size = 0
for x in range(len(itemSets)):
    size = size + len(itemSets)
print(size)
