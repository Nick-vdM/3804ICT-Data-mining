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
            if(currentNode.item == items[0]):
                currentNode.addNode(items[0])
            elif(items[0] in currentNode.children ):
                currentNode = currentNode.children[items[0]]
            else:
                currentNode.addNode(items[0])

                





