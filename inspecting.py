import sys
import csv
import numpy
from numpy import genfromtxt
import math

def readCSV(filename):
    csvArray = genfromtxt(filename, delimiter=",", dtype="U")
    return csvArray

def writeOutput(filename, data):
    with open(filename, "w") as file:
        file.write(data)

def loop(data):
    #Assign the reference label to which we will assign the value 1
    refLabel = data[1][len(data[0]) - 1]
    refCount = 0
    #Count the instances of that label in the examples (assuming binary labels)
    for row in data:
        if row[len(row) - 1] == refLabel:
            refCount = refCount + 1
    return refCount

def findEntropy(data):
    refCount = loop(data)
    #Calculate entropy
    examplesCount = len(data) - 1
    otherCount = examplesCount - refCount
    entropy = (refCount / examplesCount * math.log2(examplesCount/refCount)) + (otherCount / examplesCount * math.log2(examplesCount/otherCount))
    return entropy

def findError(data):
    refCount = loop(data)
    examplesCount = len(data) - 1
    #If the ref label wins majority vote
    if refCount / examplesCount > 0.5:
        error = (examplesCount - refCount) / examplesCount
    else:
        error = refCount / examplesCount
    return error

def main():
    inspectFile = f"handout/{sys.argv[1]}"
    outputFile = sys.argv[2]
    csvData = readCSV(inspectFile)
    entropy = findEntropy(csvData)
    error = findError(csvData)
    outputString = f"Entropy: {entropy}\nError: {error}\n"
    writeOutput(outputFile, outputString)

if __name__ == "__main__":
    main()
