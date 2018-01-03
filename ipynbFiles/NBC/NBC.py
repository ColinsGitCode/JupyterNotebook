
import csv
from unicodedata import category
import re
import string
import random
import math

# ----------------------------------------------------------------------
#  Vector Create functions
# ----------------------------------------------------------------------
def delete_strings_symbols(s):
    NoSpecialChars = s.translate ({ord(c): (" "+ c +" ") for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
    return NoSpecialChars

def remove_symbols_in_string(text,newsign=''):
    signtext = string.punctuation + newsign
    signrepl = '@'*len(signtext)
    signtable = str.maketrans(signtext,signrepl)
    return text.translate(signtable).replace('@','')

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
        count = 1
        for val in predictions:
            if val == 0:
                writer.writerow([count,"ham"])
                count += 1
            else:
                writer.writerow([count,"spam"])
                count += 1

# ----------------------------------------------------------------------
# NBC functions
# ----------------------------------------------------------------------

def split_dataset(dataset, splitRatio):
    '''Split the trianing dataset by the splitRatio'''
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
	    index = random.randrange(len(copy))
	    trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separate_by_class(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculate_probability(x, mean, stdev):
    if mean == 0 or stdev == 0:
        return 1
    else:
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateProbability_Bak(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculate_probability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculate_class_probabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def get_predictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def get_accuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def sms_dataset(splitRatio = 0.67):
    trainFileLines,trainFileDict = read_file_and_return_each_line_and_dictionary("sms_train.tsv")
    trainVectors = lines_to_vectors_Training(trainFileLines,trainFileDict)
    trainTrainingSet, trainTestSet = split_dataset(trainVectors,splitRatio)
    summaries = summarize_by_class(trainTrainingSet)
    trainPredictions = get_predictions(summaries, trainTestSet)
    trainAccuracy = get_accuracy(trainTestSet, trainPredictions)
    print("The Accuracy is: %f" %trainAccuracy)
    testFileLines,testFileDict = read_file_and_return_each_line_and_dictionary("sms_test.tsv")
    testVector = lines_to_vectors_Training(testFileLines, trainFileDict)
    newSummaries = summarize_by_class(trainVectors)
    testPredictions = get_predictions(newSummaries,testVector)
    csvFileName = "SMS_results.csv"
    write_result_in_csv(testPredictions,csvFileName)


#-------------------------------------------------------------------------------------------
# Vector Sparse Format
#-------------------------------------------------------------------------------------------

def each_line_to_vector_sparseFormat(line,dictKeysList):
    '''Return the FULL format bag-of-words vector for a line'''
    line = line.split(" ")
    dictLength = len(dictKeysList)
    vectorTemplate = {} # using dictionary
    lineVector = vectorTemplate
    for w in line:
        if w in dictKeysList:
            INDEX = dictKeysList.index(w)
            if INDEX in lineVector.keys():
                lineVector[INDEX] += 1
            else:
                lineVector[INDEX] = 1
    return lineVector


def lines_to_vectors_Training_sparseFormat(lines,dictionary):
    linesVectors = []
    listDictionary = list(dictionary.keys())
    for ele in lines:
        theVector = each_line_to_vector_sparseFormat(ele[1],listDictionary)
        if ele[0] == "ham":
            theVector = [theVector, 0]
        else:
            theVector = [theVector, 1]
        linesVectors.append(theVector)
    return linesVectors


def mean_by_postions(posList,Count):
    mean = sum(posList)/float(Count)
    return mean


def stdev_by_positions(posList,Count):
    avg =mean_by_postions(posList,Count)
    Zeros = pow(0-avg,2) * (Count-len(posList))
    NotZeros = sum([pow(x-avg,2) for x in posList])
    variance = (Zeros + NotZeros)/float(Count)
    return math.sqrt(variance)


def summarize_sparseFormat_by_class(dataset):
    vectorCountByClass = len(dataset)
    summaries_by_class = {}
    for vector in dataset:
        sparseVector = vector[0]
        keysSparseVector = sparseVector.keys()
        for pos in keysSparseVector:
            if pos in summaries_by_class.keys():
                summaries_by_class[pos].append(sparseVector[pos])
            else:
                summaries_by_class[pos] = [sparseVector[pos]]
    stats_by_class = {}
    for pos in summaries_by_class.keys():
        thePosMean = mean_by_postions(summaries_by_class[pos],vectorCountByClass)
        thePosStdev = stdev_by_positions(summaries_by_class[pos],vectorCountByClass)
        stats_by_class[pos] = (thePosMean,thePosStdev)
    return stats_by_class

def summarize_by_class_sparseFormat(dataset):
	separated = separate_by_class(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize_sparseFormat_by_class(instances)
	return summaries

def calculate_probability_sparseFormat(x, Mean, Stdev):
    if Mean == 0 or Stdev == 0:
        return 1
    else:
        exponent = math.exp(-(math.pow(x-Mean,2)/(2*math.pow(Stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * Stdev)) * exponent

def calculate_class_probabilities_sparseFormat(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in classSummaries.keys():
            mean, stdev = classSummaries[i]
            if i in inputVector[0].keys():
                x = inputVector[0][i]
                probabilities[classValue] *= calculate_probability_sparseFormat(x, mean, stdev)
            else:
                x = 0
                probabilities[classValue] *= calculate_probability_sparseFormat(x, mean, stdev)

    return probabilities

def predict_sparseFormat(summaries, inputVector):
	probabilities = calculate_class_probabilities_sparseFormat(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def get_predictions_sparseFormat(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict_sparseFormat(summaries, testSet[i])
		predictions.append(result)
	return predictions

def sms_dataset_sparseFormat(splitRatio = 0.67):
    trainLine, trainDict = read_file_and_return_each_line_and_dictionary("sms_train.tsv")
    linesVectors = lines_to_vectors_Training_sparseFormat(trainLine,trainDict)
    trainVectors, testVectors = split_dataset(linesVectors,splitRatio)
    summaries= summarize_by_class_sparseFormat(trainVectors)
    trainPredictions = get_predictions_sparseFormat(summaries, testVectors)
    trainAccuracy = get_accuracy(testVectors, trainPredictions)
    print("The Accuracy is: %f" %trainAccuracy)
    testFileLines,testFileDict = read_file_and_return_each_line_and_dictionary("sms_test.tsv")
    testVectors = lines_to_vectors_Training_sparseFormat(testFileLines, trainDict)
    newSummaries = summarize_by_class_sparseFormat(linesVectors)
    testPredictions = get_predictions_sparseFormat(newSummaries,testVectors)
    csvFileName = "SMS_results_sparseFormat.csv"
    write_result_in_csv(testPredictions,csvFileName)


def read_net_file_return_lines_and_dictionary(fileName):
    returnLine = []
    dictionary = {}
    fileLines = read_tsv_by_line(fileName)
    for line in fileLines:
        finalLine = []
        commaSplitedLine = line[0].split(",")
        if commaSplitedLine[0] == "ham":
            classFlag = "ham"
        else:
            classFlag = "spam"
        del commaSplitedLine[0]
        ipSplited = commaSplitedLine[0].split(".")
        mailSplited = commaSplitedLine[1].split("@")
        domainSplited = mailSplited[1].split(".")
        if "@" in commaSplitedLine[2]:
            mailSplited_one = commaSplitedLine[2].split("@")
            mailSplited_one[1] = mailSplited_one[1].split(">")
            domainSplited_one = mailSplited_one[1][0].split(".")
        else:
            domainSplited_one = commaSplitedLine[2]
        theStr = ""
        for ele in ipSplited:
            if ele in dictionary.keys():
                dictionary[ele] += 1
            else:
                dictionary[ele] = 1
            theStr = theStr + ele + " "
        for ele in domainSplited:
            if ele in dictionary.keys():
                dictionary[ele] += 1
            else:
                dictionary[ele] = 1
            theStr = theStr + ele + " "
        for ele in domainSplited_one:
            if ele in dictionary.keys():
                dictionary[ele] += 1
            else:
                dictionary[ele] = 1
            theStr = theStr + ele + " "
        finalLine = [classFlag, theStr]
        returnLine.append(finalLine)
    return returnLine, dictionary

def net_dataset_sparseFormat(splitRatio = 0.67):
    trainLine, trainDict = read_net_file_return_lines_and_dictionary("net_train.csv")
    linesVectors = lines_to_vectors_Training_sparseFormat(trainLine,trainDict)
    trainVectors, testVectors = split_dataset(linesVectors,splitRatio)
    summaries= summarize_by_class_sparseFormat(trainVectors)
    trainPredictions = get_predictions_sparseFormat(summaries, testVectors)
    trainAccuracy = get_accuracy(testVectors, trainPredictions)
    print("The Accuracy is: %f" %trainAccuracy)
    testFileLines,testFileDict = read_net_file_return_lines_and_dictionary("net_test.csv")
    testVectors = lines_to_vectors_Training_sparseFormat(testFileLines, trainDict)
    newSummaries = summarize_by_class_sparseFormat(linesVectors)
    testPredictions = get_predictions_sparseFormat(newSummaries,testVectors)
    #csvFileName = "NET_results_sparseFormat.csv"
    csvFileName = "XUE_net.csv"
    write_result_in_csv(testPredictions,csvFileName)


def net_dataset(splitRatio = 0.67):
    trainLine, trainDict = read_net_file_return_lines_and_dictionary("net_train.csv")
    linesVectors = lines_to_vectors_Training(trainLine,trainDict)
    trainVectors, testVectors = split_dataset(linesVectors,splitRatio)
    summaries= summarize_by_class(trainVectors)
    trainPredictions = get_predictions(summaries, testVectors)
    trainAccuracy = get_accuracy(testVectors, trainPredictions)
    print("The Accuracy is: %f" %trainAccuracy)
    testFileLines,testFileDict = read_net_file_return_lines_and_dictionary("net_test.csv")
    testVectors = lines_to_vectors_Training(testFileLines, trainDict)
    newSummaries = summarize_by_class(linesVectors)
    testPredictions = get_predictions(newSummaries,testVectors)
    csvFileName = "XUE_net_notSparse.csv"
    write_result_in_csv(testPredictions,csvFileName)

