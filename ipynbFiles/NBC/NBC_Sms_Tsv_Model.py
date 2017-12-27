
# coding: utf-8

# In[3]:


import csv
from unicodedata import category
import re
import string
import random
import math

# ----------------------------------------------------------------------
#  Datasets functions
# ----------------------------------------------------------------------
def delete_strings_symbols(s):
    NoSpecialChars = s.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
    # s = ''.join(ch for ch in s if category(ch)[0]!= 'P')
    return NoSpecialChars 

def remove_symbols_in_string(text,newsign=''):
    signtext = string.punctuation + newsign # 引入英文符号常量，可附加自定义字符，默认为空
    signrepl = '@'*len(signtext) # 引入符号列表长度的替换字符
    signtable = str.maketrans(signtext,signrepl) # 生成替换字符表
    return text.translate(signtable).replace('@','') # 最后将替换字符替换为空即可

def read_tsv_by_line(fileName):
    '''Read .tsv files line by line and return the lines as a list''' 
    '''The return lines' element are a list which contains a string '''
    lines = []
    with open(fileName,'r', encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\n')
        for row in reader:
            lines.append(row)
    return lines

def split_line_by_tab(tsvLines):
    '''Split the class in the training datasets'''
    '''Return the lines which ele[0] is CLASS, ele[1] is SMS_MESSAGES '''
    newLines = []
    EMPTY = ""
    for ele in tsvLines:
        if len(ele) == 1:
            splitEle = ele[0].split("\t")
            newLines.append(splitEle)
        else:
            splitEle = ele.split("\t")
            newLines.append(splitEle)
    return newLines

def remove_symbols_by_line(Lines):
    noSymbolLines = []
    for ele in Lines:
        ele[1] = delete_strings_symbols(ele[1])
        noSymbolLines.append(ele)
    return noSymbolLines

def update_dictionary_by_line(line,dictionary):
    '''Update the dictionary by line'''
    splitedLine =line.split(" ")
    for w in splitedLine:
        if w in dictionary.keys():
            dictionary[w] += 1
        else:
            dictionary[w] = 1
    return dictionary

def get_whole_file_dictionary(lines):
    dictionary = {}
    for ele in lines:
        dictionary = update_dictionary_by_line(ele[1],dictionary)
    return dictionary

def read_file_and_return_each_line_and_dictionary(fileName):
    '''Read a file and return the dictionary of the whole file'''
    # read file and return the context by line
    rawLines = read_tsv_by_line(fileName)
    # delete tab in each line
    splitedLines = split_line_by_tab(rawLines)
    # delete symbols in each line
    splitedLinesNoSymbol = remove_symbols_by_line(splitedLines)
    # get the dictionary
    dictionary = {}
    dictionary = get_whole_file_dictionary(splitedLinesNoSymbol)
    return splitedLinesNoSymbol, dictionary

def each_line_to_vector(line,dictKeysList):
    '''Return the FULL format bag-of-words vector for a line'''
    line = line.split(" ")
    dictLength = len(dictKeysList)
    vectorTemplate = [0] * dictLength
    lineVector = vectorTemplate
    #for i in range(dictLength):
    for w in line:
        if w in dictKeysList: 
            INDEX = dictKeysList.index(w)
            if INDEX:
                lineVector[INDEX] += 1
    return lineVector

def lines_to_vectors_Training(lines,dictionary):
    linesVectors = []
    listDictionary = list(dictionary.keys())
    for ele in lines:
        theVector = each_line_to_vector(ele[1],listDictionary) 
        if ele[0] == "ham":
            theVector.append(0)
        else:
            theVector.append(1)
        linesVectors.append(theVector)
    return linesVectors

def write_result_in_csv(predictions,csvfile):
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in predictions:
            if val == 0:
                writer.writerow(["ham"])
            else:
                writer.writerow(["spam"])
                
# ----------------------------------------------------------------------
# NBC functions
# ----------------------------------------------------------------------

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
    # 每个类中每个类的均值
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    # 每个类中每个属性的标准差（N-1）方法
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
    if mean == 0 or stdev == 0:
        return 1
    else:
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateProbability_Bak(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------

def sms_dataset():
    trainFileLines,trainFileDict = read_file_and_return_each_line_and_dictionary("sms_train.tsv")
    trainVectors = lines_to_vectors_Training(trainFileLines,trainFileDict)
    splitRatio = 0.67
    trainTrainingSet, trainTestSet = splitDataset(trainVectors,splitRatio) 
    summaries = summarizeByClass(trainTrainingSet)
    trainPredictions = getPredictions(summaries, trainTestSet)
    trainAccuracy = getAccuracy(trainTestSet, trainPredictions)
    print("The Accuracy is: %f" %trainAccuracy)
    testFileLines,testFileDict = read_file_and_return_each_line_and_dictionary("sms_test.tsv")
    testVector = lines_to_vectors_Training(testFileLines, trainFileDict) 
    testPredictions = getPredictions(summaries,testVector)
    csvFileName = "SMS_results.csv"
    write_result_in_csv(testPredictions,csvFileName) 
    
sms_dataset()
 

