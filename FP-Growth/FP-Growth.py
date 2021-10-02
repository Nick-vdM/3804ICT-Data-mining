class Node:
    def __init__(self,item,parent):
        self.parent = parent
        self.children = {}
        self.item = item
        self.count = 1

    def addNode(self,item):
        if(self.item == item):
            self.count = self.count + 1
        else:
            self.children[item] = Node(item,self)

class Tree:
    def __init__(self):
        self.root = Node(None,None)

    def insertItemSet(self,items):
        currentNode = self.root
        while(len(items) != 0):
            if(items[0] in currentNode.children):
                currentNode = currentNode.children[items[0]]
                currentNode.addNode(items[0])
                items.pop(0)
            else:
                currentNode.addNode(items[0])
                currentNode = currentNode.children[items[0]]
                items.pop(0)
            

    def printTree(self, root, level):
        print("  " * level, root.item, ":", root.count)
        for child in root.children:
            self.printTree(root.children[child],level + 1)


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
fpTree.printTree(fpTree.root,0)


        






                





