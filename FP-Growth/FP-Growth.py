from itertools import combinations

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

        if((node.count < support and node != self.root) or node.item == endItem):
            itemSet.insert(len(items), items)
            return
    
        if(node != self.root):
            items.insert(len(items),node.item)
        #print(endItem,"Adding",items)
        for key, value in node.children.items():
            self.traverseToNode(endItem, itemSet, node.children[key], support, items)

        if(len(node.children)==0):
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





def getFreqPatterns(tree, orderList):
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
        
        basePatterns = []
        conditionalTree.traverseToNode(item,basePatterns,conditionalTree.root,2,[])
        for i in range(0,len(basePatterns)):
            basePatterns[i].insert(len(basePatterns[i]),item)
            for x in range(1,len(basePatterns[i])+1):
                for subset in combinations(basePatterns[i], x):
                    if(len(subset) > 1 and item in subset):
                        freqPatterns.insert(0,subset)

    freqPatterns = list(set(freqPatterns))
    return freqPatterns
            

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


freqTable = generateFreqTable(dataSet)
#print(freqTable)

orderList = []
for i in freqTable:
    orderList.insert(len(orderList),i)
#print(orderList)

for itemSet in dataSet:
    dataSet[dataSet.index(itemSet)] = sorted(itemSet, key=lambda e: (orderList.index(e), e))

#print(dataSet)


fpTree = Tree()
for itemSet in dataSet:
    fpTree.insertItemSet(itemSet)
#fpTree.printTree(fpTree.root,0)

patterns = getFreqPatterns(fpTree, orderList)
for pattern in patterns:
    print(pattern)




                





