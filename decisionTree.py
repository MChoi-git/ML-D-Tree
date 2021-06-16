import sys
import numpy as np
from numpy import genfromtxt
import math
import csv

class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
        self.attribute = None #This should really be renamed to label

class Branch:
    def __init__(self, value):
        self.value = value
        self.next = None

def readCSV(filename):
    csvArray = genfromtxt(filename, delimiter=",", dtype="U")
    return csvArray

def calculateEntropy(data, column): #Columns = [0,1,...]
    # Assign the reference label to which we will assign the value 1
    refLabel = data[1][column]
    refCount = 0
    examplesCount = len(data) - 1
    entropy = 0
    # Count the instances of that label in the examples (assuming binary labels)
    for row in data:
        if row[column] == refLabel:
            refCount = refCount + 1
    otherCount = examplesCount - refCount
    #Calculate entropy
    if refCount and otherCount > 0:
        entropy = (refCount / examplesCount * math.log2(examplesCount / refCount)) + (otherCount / examplesCount * math.log2(examplesCount / otherCount))
    return entropy

def calculateJointEntropy(data, colX, colY): #Hardcoded for only two vars
    #Assign reference labels = 1
    refLabelX = data[1][colX]
    refLabelY = data[1][colY]
    numExamples = len(data) - 1
    countArray = [0,0,0,0] #Holds counts of X=x,Y=y pairs, ie. [(X=0,Y=0), (X=0,Y=1), (X=0,Y=1), (X=0,Y=1)]
    #Populate list with probabilities
    for row in data[1:]:
        if row[colX] != refLabelX:
            if row[colY] != refLabelY:  #X=0,Y=0
                countArray[0] += 1
            else:                       #X=0,Y=1
                countArray[1] += 1
        else:
            if row[colY] != refLabelY:  #X=1,Y=0
                countArray[2] += 1
            else:                       #X=1,Y=1
                countArray[3] += 1
    #Calculate entropy
    entropy = 0
    for entry in countArray:
        if entry > 0:
            entropy += entry / numExamples * math.log2(numExamples / entry)
    return entropy

def calculateMutualInfo(entropyX, entropyY, jointEntropyXY):
    mutualInfo = entropyX + entropyY - jointEntropyXY
    return mutualInfo

def majorityVote(data):
    labelArray = [0,0]
    print("Leaf majority vote:")
    print(f"Datapoints: {len(data) - 1}")
    print(data)
    refLabel = data[1][len(data[1]) - 1]
    print(f"Reference data is: {refLabel}")
    for row in data[1:]:
        if row[len(data[1]) - 1] == refLabel:
            labelArray[0] += 1
        else:
            labelArray[1] += 1
            otherLabel = row[len(data[1]) - 1]
    print(f"Label array: {labelArray}")
    if labelArray.index(max(labelArray)) == 0:
        print(f"Majority vote returned {refLabel}")
        return refLabel
    else:
        print(f"Majority vote returned {otherLabel}")
        return otherLabel
1
def stumpify(root, maxDepth, currentDepth):
    currentDepth += 1
    #Stopping condition for max depth
    if maxDepth <= currentDepth:
        label = majorityVote(root.data)
        root.data = label
        return root
    mutualInfoArray = [] #Each entry in list corresponds to each attribute
    #Find best possible attribute via splitting criterion mutual info
    for attribute in range(0, len(root.data[0]) - 1): #Loop through possible attributes to split on
        labelColumn = len(root.data[0]) - 1
        entropyX = calculateEntropy(root.data, attribute)
        entropyY = calculateEntropy(root.data, labelColumn)
        jointEntropyXY = calculateJointEntropy(root.data, attribute, labelColumn)
        mutualInfoXY = calculateMutualInfo(entropyX, entropyY, jointEntropyXY)
        print(f"Mutual info calculation for: {attribute}\nH(X):   {entropyX}\nH(Y):   {entropyY}\nH(X,Y): {jointEntropyXY}\nI(X,Y): {mutualInfoXY} <==")
        mutualInfoArray.append(mutualInfoXY)
    bestAttribute = mutualInfoArray.index(max(mutualInfoArray)) #The best attribute is the one with the highest mutual info with the labels
    print(f"Best mutual info is: {mutualInfoArray[bestAttribute]}")
    if mutualInfoArray[bestAttribute] <= 0: #Stopping criterion for leaf
        label = majorityVote(root.data)
        root.attribute = label
        return root
    print(f"Splitting on best attirbute: {bestAttribute}")
    root.attribute = root.data[0][bestAttribute]
    #Partition dataset according to best attribute
    refAttributeValue = root.data[1][bestAttribute] #Save the reference attribute value to which we assign 0
    print(f"Reference attribute value: {refAttributeValue}")
    refAttributeIndexArray = [0]
    nonRefAttributeIndexArray = [0]
    for i in range(1,len(root.data),1): #Partition data on best attribute
        currentEntry = root.data[i][bestAttribute] #For readability
        currentEntryRow = root.data[i]
        if currentEntry == refAttributeValue:   #If the current entry matches the attribute
            refAttributeIndexArray += [i]  #Save the row index into the partition array
        else:   #If the current entry doesn't match
            nonRefAttributeIndexArray += [i]    #Save the row index into the other partition array
    refAttributeValueArray = root.data[np.ix_(refAttributeIndexArray)]  #Pull the rows from root data np array into paritioned np arrays
    nonRefAttributeValueArray = root.data[np.ix_(nonRefAttributeIndexArray)]
    nonRefAttributeValue = nonRefAttributeValueArray[0][bestAttribute] #Save the non reference attribute value for later use
    print(f"Partition 0: {refAttributeValueArray}")
    print(f"Partition 1: {nonRefAttributeValueArray}")
    #Branch into new nodes
    #if sum(mutualInfoArray) > 0:
    root.left = Branch(refAttributeValue)
    root.left.next = stumpify(Node(refAttributeValueArray), maxDepth, currentDepth) #Recurse left
    root.right = Branch(nonRefAttributeValue)
    root.right.next = stumpify(Node(nonRefAttributeValueArray), maxDepth, currentDepth) #Recurse right
    return root

    #Temporary return for testing
    '''returnArray = [bestAttribute]
    returnArray += [refAttributeValue]
    returnArray.append(refAttributeValueArray)
    returnArray.append(nonRefAttributeValueArray)
    return returnArray'''

def train(dataset, maxDepth):
    print("Stumpifying")
    root = Node(dataset)    #Put all the data into the root node
    return stumpify(root, maxDepth, 0)

def prettyPrint(root):
    print(root.data)
    print(root.attribute)
    if root.left is not None and root.left.next is not None:
        print(root.left.value)
        prettyPrint(root.left.next)
    if root.right is not None and root.right.next is not None:
        print(root.right.value)
        prettyPrint(root.right.next)

def main():
    print("Tree Start!")
    #Get the input csv and output file from cl
    trainInput = f"handout/{sys.argv[1]}"    # Path to training input csv file
    testInput = f"handout/{sys.argv[2]}"    # Path to test input csv file
    '''
    maxDepth = sys.argv[3]      # Maximum depth to which the tree should be built
    trainOut = sys.argv[4]      # Path of output .labels file to which the predictions on the training data should be written
    testOut = sys.argv[5]       # Path of output .labels file to which the predictions on the test data should be written
    metricsOut = sys.argv[6]    # Path of the output .txt file to which metrics such as train and test error should be written
    '''
    #Store training and test sets to numpy arrays
    trainingData = readCSV(trainInput)
    testData = readCSV(testInput)

    #Test the entropy function
    print("Testing entropy calculations:")
    print(100 * '*')
    print(f"Training file: {trainInput} | Test file: {testInput}")
    print(100 * '*')

    trainYEntropy = calculateEntropy(trainingData, len(trainingData[0]) - 1)
    print(f"Training entropy: {trainYEntropy}")

    trainXYJointEntropy = calculateJointEntropy(trainingData, 0, len(trainingData[0]) - 1) #Test these with first column and last column
    print(f"Training Joint entropy: {trainXYJointEntropy}")

    testYEntropy = calculateEntropy(testData, len(testData[0]) - 1)
    print(f"Test entropy: {testYEntropy}")

    testXYJointEntropy = calculateJointEntropy(testData, 0, len(testData[0]) - 1) #Test these with first column and last column
    print(f"Test Joint entropy: {testXYJointEntropy}")

    trainXEntropy = calculateEntropy(trainingData, 0)
    trainMutualInfo = calculateMutualInfo(trainYEntropy, trainXEntropy, trainXYJointEntropy)
    print(f"Training mutual info (attr1, Y): {trainMutualInfo}")

    testXEntropy = calculateEntropy(testData, 0)
    testMutualInfo = calculateMutualInfo(testYEntropy, testXEntropy, testXYJointEntropy)
    print(f"Training mutual info (attr1, Y): {testMutualInfo}")

    decisionTree = train(trainingData, 4)

    '''print(100*'*')
    print("Testing stump function:")
    infoArray = train(trainingData,1)
    for item in infoArray:
        print(type(item))
        print(item)'''
    print("Printing tree...")
    prettyPrint(decisionTree)

if __name__ == "__main__":
    main()