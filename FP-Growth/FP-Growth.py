import itertools
import pickle_manager



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
    def __init__(self, itemSetList, frequency, minSupport):
        self.root = Node(None,None)
        self.headerTable = {}
        self.createTree(itemSetList, frequency, minSupport)

    def createTree(self,itemSetList,frequency, minSupport):
        #Create header table with each item and count frequcny of each
        i = 0
        for itemSet in itemSetList:
            for item in itemSet:
                if(item in self.headerTable):
                    self.headerTable[item] = [self.headerTable[item][0] + frequency[i], None]
                else:
                    self.headerTable[item] = [frequency[i], None]
            i=i+1

        for k in list(self.headerTable.keys()):
            if self.headerTable[k][0] < minSupport:
                del self.headerTable[k]

        i = 0
        for itemSet in itemSetList:
            itemSet = [item for item in itemSet if item in self.headerTable]
            itemSet.sort(key=lambda item: self.headerTable[item][0], reverse=True)
            self.insertItemSet(itemSet, frequency[i])
            i=i+1


    def insertItemSet(self,items,count):
        currentNode = self.root
        while(len(items) != 0):
            if(items[0] in currentNode.children):
                currentNode = currentNode.children[items[0]]
                currentNode.addNode(items[0], count)
                items.pop(0)
            else:
                currentNode.addNode(items[0], count)
                #If item already in the header table then follow the links to the next slot
                currentItem = self.headerTable[items[0]][1] 
                if(currentItem == None):
                    self.headerTable[items[0]][1] = currentNode.children[items[0]]
                else:
                    while(1):
                        if(currentItem.link == None):
                            currentItem.link = currentNode.children[items[0]]
                            break
                        else:   
                            currentItem = currentItem.link   
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

    def getAllPaths(self, item):
        allPaths = []
        currentNode = self.headerTable[item][1]
        while(1):
            branch = []
            self.traverseToRoot(currentNode,branch)
            branch.remove(item)
            if(currentNode.link == None):
                if(len(branch) > 0):
                    allPaths.insert(0,(branch,currentNode.count))
                break
            else:
                if(len(branch) > 0):
                    allPaths.insert(0,(branch,currentNode.count))
                currentNode = currentNode.link
        return allPaths



def mineTree(tree,preFix,freqItemsList,minSupport):
    sortedItemList = [item[0] for item in sorted(list(tree.headerTable.items()), key=lambda p:p[1][0])] 
    #For each item from least supported to most
    for item in sortedItemList:
        newFreqSet = preFix.copy()
        newFreqSet.add(item)
        freqItemsList.append(newFreqSet)
        allPaths = tree.getAllPaths(item)
        frequency = []
        condPaths = []
        for path in allPaths:
            frequency.insert(len(frequency),path[1])
            condPaths.insert(len(condPaths),path[0])
        #Create new tree
        condTree = Tree(condPaths,frequency,minSupport)
        if len(condTree.headerTable) > 0:
            mineTree(condTree,newFreqSet, freqItemsList, minSupport)


def getAssociationRules(freqItemsList, dataSet, minConfidence):
    rules = []
    for items in freqItemsList:
        items = list(items)
        if(len(items) < 2):
            continue
        AUBsupport = 0
        for itemSet in dataSet:
            if(len(itemSet) > 0):
                if(all(item in itemSet for item in items)):
                    AUBsupport = AUBsupport + 1
        for L in range(0, len(items)):
            for subset in itertools.combinations(items, L):
                A = list(subset)
                B = [x for x in items if x not in A]
                Asupport = 0
                if(len(A) > 0 and len(B) > 0):
                    for itemSet in dataSet:
                        if(len(itemSet) > 0):
                            if(all(item in itemSet for item in A)):
                                Asupport = Asupport + 1        
                if(Asupport > 0):
                    if(AUBsupport/Asupport > minConfidence):
                        rules.insert(0,[A,"->",B,AUBsupport/Asupport])
    return rules


df = pickle_manager.load_pickle("pickles\organised_ratings.pickle.lz4")

for y in range(1,11):
    supportPercentage = 0.01 * y
    for x in range(1,6):
        print("Movie Rating: ", x)
        print("Support Percentage", supportPercentage)
        dataSet = []
        for i in range(1,df['userId'].max()+1):
            itemSet = []
            itemSet = df.loc[(df['rating'] == x) & (df['userId'] == i)]['imdbId'].tolist()
            dataSet.insert(0,itemSet)

        dataSetSize = len(dataSet)
        frequency = [1 for i in range(len(dataSet))]
        fpTree = Tree(dataSet,frequency,supportPercentage*dataSetSize)
        freqItemsList = []
        mineTree(fpTree, set(), freqItemsList,supportPercentage*dataSetSize)
        print("Number of frequent sets:", len(freqItemsList))
        if(len(freqItemsList) > 0):
            rules = getAssociationRules(freqItemsList, dataSet, 0.8)
            print("Number of rules: ", len(rules))