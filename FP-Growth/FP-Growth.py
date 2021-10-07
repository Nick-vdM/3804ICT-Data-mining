from itertools import combinations, count
from time import perf_counter

from numpy import fromregex
from pandas.core import base
import pickle_manager
import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import sys

class Node:
    def __init__(self,item,parent,count = 1):
        self.parent = parent
        self.children = {}
        self.item = item
        self.count = count
        self.link = None

    def addNode(self,item, count = 1):
        if(self.item == item):
            self.count = self.count + count
        else:
            self.children[item] = Node(item,self, count)

class Tree:
    def __init__(self):
        self.root = Node(None,None)
        self.headerTable = {}

    def insertItemSet(self,items, count = 1):
        currentNode = self.root
        while(len(items) != 0):
            if(items[0] in currentNode.children):
                currentNode = currentNode.children[items[0]]
                currentNode.addNode(items[0], count)
                items.pop(0)
            else:
                currentNode.addNode(items[0], count)
                if(items[0] in self.headerTable):
                    currentItem = self.headerTable[items[0]] 
                    while(1):
                        if(currentItem.link == None):
                            currentItem.link = currentNode.children[items[0]]
                            break
                        else:   
                            currentItem = currentItem.link   
                else:
                    self.headerTable[items[0]] = currentNode.children[items[0]]

                currentNode = currentNode.children[items[0]]
                items.pop(0)

    def traverseToRoot(self,node,branch):
        branch.insert(0,node.item)
        if(node.parent == self.root):
            return branch
        else:
            return self.traverseToRoot(node.parent,branch)

            
    def printTree(self, root, level):
        print("  " * level, root.item, ":", root.count)
        for child in root.children:
            self.printTree(root.children[child],level + 1)

    def traverseToNode(self,endItem,itemSet, node, support, parentItems):
        items = parentItems.copy()
        #print(endItem,"Giving",items)

        if((node.count < support and node != self.root) or (node.item == endItem)):
            if(len(items) > 0):
                itemSet.insert(len(itemSet), items)
            return

        if(node != self.root):
            items.insert(len(items),[node.item,node.count])

        #print(endItem,"Adding",items)
        for key, value in node.children.items():
            self.traverseToNode(endItem, itemSet, node.children[key], support, items)

        if(len(node.children)==0 and len(items) > 0):
            itemSet.insert(len(items), items)
            return


def generateFreqTable(dataSet):
    freqTable = {}
    for itemSet in dataSet:
        for item in itemSet:
            if(item in freqTable):
                freqTable[item] = freqTable[item] + 1
            else:
                freqTable[item] = 1

    freqTable = dict(sorted(freqTable.items(), key=lambda item: -item[1]))
    return freqTable





def getFreqPatterns(tree, orderList, support, freqTable, dataSetSize):
    freqPatterns = []
    for item in reversed(orderList):
        currentNode = tree.headerTable[item]
        prefixPaths = []
        while(1):
            branch = tree.traverseToRoot(currentNode,[])
            branch.remove(item)
            prefixPaths.insert(len(prefixPaths),[branch,currentNode.count])
            if(currentNode.link == None):
                break
            else:
                currentNode = currentNode.link

        conditionalTree = Tree()
        for prefixPath in prefixPaths:
            conditionalTree.insertItemSet(prefixPath[0],prefixPath[1])
        #print(item, support)
        #conditionalTree.printTree(conditionalTree.root,0)
        basePatterns = []
        conditionalTree.traverseToNode(item,basePatterns,conditionalTree.root,support,[])
        if(item == 'tt0076759'):
            conditionalTree.printTree(conditionalTree.root,0)
        for i in range(0,len(basePatterns)):
            basePatterns[i].insert(len(basePatterns[i]),item)
            for x in range(0,len(basePatterns[i])+1):
                for subset in combinations(basePatterns[i], x):
                    if(item in subset):
                        freqPatterns.insert(0,subset)

        if(len(basePatterns) == 0):
            freqPatterns.insert(0,[item])

    for x in range(0,len(freqPatterns)):
        supportCount = float("inf")
        if(isinstance(freqPatterns[x], tuple)):
            freqPatterns[x] = list(freqPatterns[x])
        if(len(freqPatterns[x]) > 1):
            for i in range(0,len(freqPatterns[x])):
                if(isinstance(freqPatterns[x][i], list)):
                    if(freqPatterns[x][i][1] < supportCount):
                        supportCount = freqPatterns[x][i][1]
                    freqPatterns[x][i] = freqPatterns[x][i][0]
            freqPatterns[x].insert(len(freqPatterns[x]),supportCount)
        else:
            freqPatterns[x].insert(len(freqPatterns[x]), freqTable[freqPatterns[x][0]])
   
    #Remove single items that dont have enough support
    for elem in list(freqPatterns):
        if len(elem) == 2 and elem[1] < support:
            freqPatterns.remove(elem)
    #Convert support count to percentage
    for i in range(0,len(freqPatterns)):
        freqPatterns[i][-1] =  freqPatterns[i][-1]/dataSetSize

    return freqPatterns


df = pickle_manager.load_pickle("pickles\organised_ratings.pickle.lz4")


dataSet = []
for i in range(1,df['userId'].max()+1):
    itemSet = []
    itemSet = df.loc[(df['rating'] == 5) & (df['userId'] == i)]['imdbId'].tolist()
    dataSet.insert(0,itemSet)

            
"""
dataSet = [ ['I1','I2','I5'],
            ['I2','I4'],
            ['I2','I3'],
            ['I1','I2','I4'],
            ['I1','I3'],
            ['I2','I3'],
            ['I1','I3'],
            ['I1','I2','I3','I5'],
            ['I1','I2','I3'],
        ]
"""

dataSetSize = len(dataSet)

print("Generating Frequency table")
start = perf_counter()
freqTable = generateFreqTable(dataSet)
#print(freqTable)

orderList = []
for i in freqTable:
    orderList.insert(len(orderList),i)
#print(orderList)

for itemSet in dataSet:
    dataSet[dataSet.index(itemSet)] = sorted(itemSet, key=lambda e: (orderList.index(e), e))

#print(dataSet)

print("Creating FP tree")
fpTree = Tree()
for itemSet in dataSet:
    fpTree.insertItemSet(itemSet)
#fpTree.printTree(fpTree.root,0)

supportPercentage = 0.05
print("Mining frequent patterns")
print("Support of ",int(round(supportPercentage*dataSetSize)) )
patterns = getFreqPatterns(fpTree, orderList, int(round(supportPercentage*dataSetSize)), freqTable, dataSetSize)
end = perf_counter()
print(end - start, start, end)

for pattern in patterns:
    print(pattern)
print(len(patterns))



                



