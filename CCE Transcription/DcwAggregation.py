
import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as mplplot
import dateutil.parser
import pickle
import sys
import re as regex
import copy
import itertools
import functools
import gc
import os
import glob
from collections import Counter, OrderedDict

# Classes to encapsulate single lines of text and multiple lines of text
from TextLine import TextLine
from TelegramLines import TelegramLines
from LineMatcher import LineMatcher
from MetaTagState import MetaTagState
from StatefulWord import StatefulWord

verbose = True
extraVerbose = False
if verbose:
    import pprint
    # from IPython.core.display import display
    pprinter = pprint.PrettyPrinter(indent=4)

# get_ipython().magic('matplotlib inline')

# ## Globally relevant variables

saveIdentifiedLineDetails = True  # Need to do this to recompute for every codebook
applyDoubleLineFix = False
applyDoubleWordFilter = True
applyDoubleLineFilter = True
classificationBaseDirectory = '/Users/stellachoi/Documents/SDL/dcw/dcw_file_test'
databaseNamePattern = 'dcwConsensus_{mss_label}'
aggregatedDataFileNamePattern = 'decoding-the-civil-war-aggregated_{mss_label}.txt'
# 'decoding-the-civil-war-consensus-linewise.csv'
aggregatedDataCsvFileNamePattern = 'decoding-the-civil-war-consensus-linewise_{mss_label}.csv'
# 'decoding-the-civil-war-consensus-subjectwise.csv'
aggregatedDataSubjectWiseCsvFileNamePattern = 'decoding-the-civil-war-consensus-subjectwise_{mss_label}_withBreaks.csv'
# 'dataWithLineIDs_subset.pkl'
identifiedLineFilePathPattern = 'dataWithLineIDs_subset_{mss_label}.pkl'
liveDate = dateutil.parser.parse("2016-06-20T00:00:00.00Z")
workflow_id = "5925"

def loadSubjectData(subjectDataFileName):
    subject_data = []
    #subjectColumns = ['subject_id', 'project_id', 'workflow_id', 'subject_set', 'metadata', 'locations']
    with open(subjectDataFileName) as csvfile:
        parsedSubjectCsv = csv.DictReader(csvfile)
        for subject in parsedSubjectCsv:
            parsedLocations = json.loads(subject['locations'])
            parsedMetaData = json.loads(subject['metadata'])
            if subject['workflow_id'] != workflow_id:
                continue
            subject_data.append({
                'subject_id': int(subject['subject_id']),
                'img_id': parsedMetaData['id'],
                'url': parsedLocations['0']
            })

    subjectsFrame = pd.DataFrame.from_records(subject_data, index='subject_id')

    return subjectsFrame

# Parse the downloaded classification data into data structures for processing
def loadTelegrams(sampleDataFileName):

    titles = {} #for t1
    publishdates = {} #for t6
    copyrights = {} #for t8

    with open(sampleDataFileName) as csvfile:
        parsedCsv = csv.DictReader(csvfile)
        nTask1Parsed = 0
        nTask8Parsed = 0
        for recordIndex, record in enumerate(parsedCsv):
            done = False
            isTask1Exist = False
            isTask8Exist = False
            isTask6Exist = False

            # check the date that the classification was made
            if "metadata" in record:
                parsedMetadata = json.loads(record["metadata"])
                parsedDate = dateutil.parser.parse(
                    parsedMetadata['started_at'])
                # skip "testing" data before the site went live
                if parsedDate < liveDate:
                    continue

            # parse the annotations and the subject data
            parsedAnnotations = json.loads(record["annotations"])
            parsedSubjectData = json.loads(record["subject_data"])

            # initialize container for transcribed lines
            titlesLines = TelegramLines()
            copyrightLines = TelegramLines()
            publishdateLines = TelegramLines()


            # loop over tasks in the annotation
            for task in parsedAnnotations:
                # Check if the current record is for a telegram (tasks may be stored out of order, so
                # some tasks may be processed before non-telegrams are caught -
                # inefficient but unavoidable?)
                task_values = []
                if task['task'] == "T0" and len(task['value']) > 0:
                    task_values.append(task['value'])
                elif task['task'] == "T1" and len(task['value']) > 0:
                    isTask1Exist = True
                    # process the lines that were transcribed for this task
                    for taskValueItem in task['value']:
                        titleLine = TextLine(
                            taskValueItem['x1'], taskValueItem['y1'],
                            taskValueItem['x2'], taskValueItem['y2'],
                            taskValueItem['details'][0]['value'])
                        titlesLines.addLine(titleLine)

                elif task['task'] == "T6" and len(task['value']) > 0:
                    isTask6Exist = True
                    # process the lines that were transcribed for this task
                    for taskValueItem in task['value']:
                        publishdateLine = TextLine(
                            taskValueItem['x1'], taskValueItem['y1'],
                            taskValueItem['x2'], taskValueItem['y2'],
                            taskValueItem['details'][0]['value'])
                        publishdateLines.addLine(publishdateLine)

                elif task['task'] == "T8" and len(task['value']) > 0:
                    isTask8Exist = True
                    # process the lines that were transcribed for this task
                    for taskValueItem in task['value']:
                        copyrightLine = TextLine(
                            taskValueItem['x1'], taskValueItem['y1'],
                            taskValueItem['x2'], taskValueItem['y2'],
                            taskValueItem['details'][0]['value'])
                        copyrightLines.addLine(copyrightLine)

            # if the transcribed lines of a telegram have been processed then update the
            # list of independent transcriptions for this subject
            if isTask1Exist:
                nTask1Parsed += 1
                new_list = titles.get(int(record['subject_ids']), [])
                new_list.append((recordIndex, titlesLines))
                titles[int(record['subject_ids'])] = new_list

            if isTask6Exist:
                new_list = copyrights.get(int(record['subject_ids']), [])
                new_list.append((recordIndex, publishdateLines))
                publishdates[int(record['subject_ids'])] = new_list

            if isTask8Exist:
                nTask8Parsed += 1
                new_list = copyrights.get(int(record['subject_ids']), [])
                new_list.append((recordIndex, copyrightLines))
                copyrights[int(record['subject_ids'])] = new_list

    return titles, copyrights, publishdates, nTask1Parsed, nTask8Parsed

# Cast parsed data into structures that enable "straightfoward"
# aggregation analysis
def processLoadedTelegrams(telegrams):

    transcriptionLineStats = {}
    transcriptionLineDetails = []
    print(telegrams)
    # loop over distinct subjects (currently individual telegram-type pages,
    # codebook handling to be implemented)
    for key, transcriptions in telegrams.items():
        totalLines = 0
        maxLines = 0
        minLines = sys.maxsize
        # loop over individual transcriptions of the subject
        for transcriptionData in transcriptions:
            transcription = transcriptionData[1]
            transcriptionIndex = transcriptionData[0]
            # process overall transcription statistics for this subject
            numLines = transcription.getNumLines()
            totalLines += numLines
            maxLines = numLines if numLines > maxLines else maxLines
            minLines = numLines if numLines < minLines else minLines
            # process the lines of the individual transcriptions of a subject
            for textLine in transcription.getLines():
                # Add a dictionary describing the current line
                lineDescription = {
                    'subjectKey': key,
                    'transcriptionIndex': transcriptionIndex,
                    'numLines': numLines,
                    'x1': textLine.getStart()['x'],
                    'y1': textLine.getStart()['y'],
                    'x2': textLine.getEnd()['x'],
                    'y2': textLine.getEnd()['y'],
                    'words': textLine.getWords()
                }
                transcriptionLineDetails.append(lineDescription)
        transcriptionLineStats.update({
            key: {
                'minLines': minLines,
                'maxLines': maxLines,
                'meanLines': totalLines / float(len(transcriptions))
            }
        })

    transcriptionLineDetailsFrame = pd.DataFrame(data=transcriptionLineDetails)
    transcriptionLineDetailsIndex = pd.MultiIndex.from_arrays([
        transcriptionLineDetailsFrame['subjectKey'],
        transcriptionLineDetailsFrame['y1'],
        transcriptionLineDetailsFrame['y2'],
        transcriptionLineDetailsFrame['x1'],
        transcriptionLineDetailsFrame['x2']
    ])
    transcriptionLineDetailsFrame = transcriptionLineDetailsFrame.set_index(
        transcriptionLineDetailsIndex)
    transcriptionLineDetailsFrame = transcriptionLineDetailsFrame.sort_index(
        level=0, sort_remaining=True)

    return transcriptionLineStats, transcriptionLineDetailsFrame


# Group transcriptions of individual lines according to spatial proximity
# NOTE: that a very important parameter for this process is the pixel
# tolerance that specifies the allowed disparity between corresponding *y*
# coordinates of separately marked annotated lines
def groupTranscriptionsLinewise(transcriptionLineDetailsFrame, lineTolerance=40, identifiedLineFilePath=None, saveIdentifiedLineDetails=False):

    if saveIdentifiedLineDetails or identifiedLineFilePath is None:
        transcriptionLineDetailsFrame['bestLineIndex'] = pd.Series(
            np.zeros_like(transcriptionLineDetailsFrame['subjectKey']),
            index=transcriptionLineDetailsFrame.index)
        # iterate over rows in sorted, grouped dataset and insert the best line
        # index
        bestLineIndex = -1
        currentSubject = -1
        numSubjectsProcessed = 0
        # line matcher with 40 pixel tolerance for y coordinates of lines that are
        # considered to be the same
        lineMatcher = LineMatcher(lineTolerance)
        for index, row in transcriptionLineDetailsFrame.iterrows():
            # if this is a new subject, reset the line index
            if currentSubject != index[0]:
                bestLineIndex = -1
                currentSubject = index[0]
                numSubjectsProcessed += 1
                if numSubjectsProcessed % 100 == 0:
                    print('Processed {0} subjects...'.format(
                        numSubjectsProcessed))

            # if the line coordinates do not match within tolerance, then increment
            # the line index
            if lineMatcher.compare((index[3], index[1], index[4], index[2])):
                bestLineIndex += 1

            # update the dataframe with the best line index
            transcriptionLineDetailsFrame.ix[index,
                                             'bestLineIndex'] = bestLineIndex
    else:
        identifiedLineFile = open(identifiedLineFilePath, 'rb')
        transcriptionLineDetailsFrame = pickle.load(identifiedLineFile)
        identifiedLineFile.close()

    return transcriptionLineDetailsFrame

# display(transcriptionLineDetailsFrame)
# pprinter.pprint(transcriptionLineStats)

# subjectsFrame.loc[1960106, 'url'].iloc[3]

# Experimental: Evaluate first pass line consensus
# Combine adjacent line groups that have very close mean Y values.

def doubleLineFix(transcriptionLineDetailsFrame, applyDoubleLineFix=False):
    if applyDoubleLineFix:
        transcriptionLineDetailsFirstPass = transcriptionLineDetailsFrame.set_index(
            'bestLineIndex', drop=False, append=True)
        transcriptionLineDetailsFirstPass[
            'oldBestLineIndex'] = transcriptionLineDetailsFirstPass[
                'bestLineIndex'].astype(np.int64)
        transcriptionLineDetailsFirstPass = transcriptionLineDetailsFirstPass.groupby(
            level=[0, 5]).aggregate({
                'y1': np.mean,
                'y2': np.mean,
                'bestLineIndex': 'first',
                'oldBestLineIndex': 'first'
            })
        transcriptionLineDetailsFirstPass['meanY'] = 0.5 * (
            transcriptionLineDetailsFirstPass['y1'] +
            transcriptionLineDetailsFirstPass['y2'])
        transcriptionLineDetailsFirstPass.set_index(
            'meanY', drop=False, append=True)

        oldColumnNames = transcriptionLineDetailsFirstPass.columns
        newColumnNames = [(name if name != 'bestLineIndex' else 'newBestLineIndex')
                          for name in oldColumnNames]
        transcriptionLineDetailsFirstPass.columns = newColumnNames

        thisRow = transcriptionLineDetailsFirstPass.iloc[0]
        # display(thisRow)

        meanYThreshold = 20
        bestLineIndexDecrement = 0

        for index, nextRow in transcriptionLineDetailsFirstPass.iterrows():
            if nextRow['newBestLineIndex'] == 0:  # new subject
                bestLineIndexDecrement = 0  # so reset the decrement
            elif np.abs(thisRow['meanY'] - nextRow['meanY']) < meanYThreshold:
                bestLineIndexDecrement += 1

            transcriptionLineDetailsFirstPass.loc[
                index, 'newBestLineIndex'] -= int(bestLineIndexDecrement)

            thisRow = nextRow

        transcriptionLineDetailsFirstPass.reset_index(
            level='bestLineIndex', drop=True, inplace=True)
        transcriptionLineDetailsFirstPass.columns = oldColumnNames

        transcriptionLineDetailsSecondPass = transcriptionLineDetailsFirstPass.set_index(
            'bestLineIndex', drop=False, append=True)
        transcriptionLineDetailsMismatches = transcriptionLineDetailsSecondPass[(
            transcriptionLineDetailsSecondPass['bestLineIndex'] !=
            transcriptionLineDetailsSecondPass['oldBestLineIndex'])]
        transcriptionLineDetailsMismatches.reset_index(
            level='bestLineIndex', drop=True, inplace=True)
        display(transcriptionLineDetailsMismatches)
        for mismatchIndex, mismatchRow in transcriptionLineDetailsMismatches.iterrows(
        ):
            for index, row in transcriptionLineDetailsFrame.loc[
                    mismatchIndex].iterrows():
                if row['bestLineIndex'] == mismatchRow['oldBestLineIndex']:
                    transcriptionLineDetailsFrame.loc[
                        mismatchIndex, 'bestLineIndex'] = mismatchRow[
                            'bestLineIndex']

    return transcriptionLineDetailsFrame


def computeConsensusWordReliability(wordOptions):
    uniqueWordOptions = list(set(wordOptions))
    # simple logic to return 0 reliability for words with very few consistent
    # transcriptions
    if len(wordOptions) < 2:
        return -0.25

    if (len(wordOptions) < 3 and len(uniqueWordOptions) > 1):
        return -0.5

    # more complicated logic that computes the fraction of transcribed words that equal the
    # consensus
    wordCounter = Counter(wordOptions)
    consensusWord, consensusWordCount = wordCounter.most_common(1)[0]
    return consensusWordCount / float(len(wordOptions))


# Now aggregate the text of the spatially matched lines
# In the process, identify, strip and note any metatags e.g.
# `[unclear][/unclear]` that surround individual words.

def aggregateSentences(sentences):
    metaTagState = MetaTagState()

    unclearPattern = r'(\[unclear\]).+?(\[/unclear\])'
    insertionPattern = r'(\[insertion\]).+?(\[/insertion\])'
    deletionPattern = r'(\[deletion\]).+?(\[/deletion\])'

    emptyTagPairPattern = r'\[([^/]+?)\]\[/\1\]'

    genericStartPattern = r'(\[([^/]+?)\])'
    genericEndPattern = r'(\[/(.+?)\])'

    aggregatedSentence = {
        'reliability': 0.0,
        'wordReliabilities': [],
        'words': []
    }
    statefulAggregatedSentence = {
        'reliability': 0.0,
        'wordReliabilities': [],
        'words': []
    }
    iSentence = 0

    for sentence in sentences:

        print("aggregate sentence", sentence)
        # metatag pairs are better described in "sentence coordinates", these
        # can always be mapped to words later
        fullSentence = ' '.join(sentence)

        # Remove any empty metatag pairs
        sentenceLength = len(sentence)
        while True:
            fullSentence = regex.sub(emptyTagPairPattern, '', fullSentence)
            if len(fullSentence) == sentenceLength:
                # no further replacement possible
                break
            else:
                sentenceLength = len(fullSentence)

        unclearResults = regex.finditer(unclearPattern, fullSentence)

        insertionResults = regex.finditer(insertionPattern, fullSentence)

        deletionResults = regex.finditer(deletionPattern, fullSentence)

        for unclearResult in unclearResults:
            '''print ('\nUnclear:\n', sentence, '\nnumgroups (start)', len(list(unclearResult.groups())), 'groups => ', list(unclearResult.groups()))
            for index, match in enumerate(unclearResult.groups()) :
                print (match, unclearResult.start(
                    index+1), unclearResult.end(index+1))
            print ('Tagging unclear between' , unclearResult.end(1), 'and', unclearResult.start(2))'''
            metaTagState.setTag('unclear',
                                unclearResult.end(1), unclearResult.start(2))

        for insertionResult in insertionResults:
            '''print ('\nInsertion:\n', sentence, '\nnumgroups (start)', len(list(insertionResult.groups())), 'groups => ', list(insertionResult.groups()))
            for index, match in enumerate(insertionResult.groups()) :
                print (match, insertionResult.start(index+1), insertionResult.end(index+1))'''
            metaTagState.setTag('insertion',
                                insertionResult.end(1),
                                insertionResult.start(2))

        for deletionResult in deletionResults:
            '''print ('\nDeletion:\n', sentence, '\nnumgroups (start)', len(list(deletionResult.groups())), 'groups => ', list(deletionResult.groups()))
            for index, match in enumerate(deletionResult.groups()) :
                print (match, deletionResult.start(index+1), deletionResult.end(index+1))'''
            metaTagState.setTag('deletion',
                                deletionResult.end(1), deletionResult.start(2))

        iWord = 0
        sentencePosition = 0
        for word in sentence:
            if (len(aggregatedSentence['words']) < iWord + 1):
                aggregatedSentence['words'].append([])
                statefulAggregatedSentence['words'].append([])

            nonMetaWord = regex.sub(genericStartPattern, '', word)
            nonMetaWord = regex.sub(genericEndPattern, '', nonMetaWord)
            if len(nonMetaWord) > 0:
                aggregatedSentence['words'][iWord].append(nonMetaWord)
                statefulAggregatedSentence['words'][iWord].append(
                    StatefulWord(nonMetaWord, (
                        sentencePosition, sentencePosition + len(word)
                    ), copy.deepcopy(metaTagState.getSetTags()), sentence))
                # Only increment wordcount if there was actually a word and not
                # just a collection of metatags
                iWord += 1
            sentencePosition += len(word) + 1

        metaTagState.reset()
        iSentence += 1
        # END LOOP OVER TRANSCRIBED SENTENCES ASSOCIATED WITH THIS LINE

    for wordOptions, statefulWordOptions in zip(
            aggregatedSentence['words'], statefulAggregatedSentence['words']):
        wordOptions.sort(key=Counter(wordOptions).get, reverse=True)
        statefulWordOptions.sort(
            key=Counter(statefulWordOptions).get, reverse=True)
        # Compute the reliability of each word's consensus transcription
        statefulAggregatedSentence['wordReliabilities'].append(
            computeConsensusWordReliability(wordOptions))
        # The word reliability can be outside the 0-1 range in special cases, so adjust clamp
        # those values appropriately
        clampedWordReliability = statefulAggregatedSentence[
            'wordReliabilities'][-1]
        clampedWordReliability = clampedWordReliability if clampedWordReliability >= 0.0 else 0.0
        clampedWordReliability = clampedWordReliability if clampedWordReliability <= 1.0 else 1.0

        statefulAggregatedSentence['reliability'] += clampedWordReliability
    try:
        statefulAggregatedSentence['reliability'] /= float(
            len(statefulAggregatedSentence['words']))
    except Exception as e:
        # print ('Exception on {} : {}'.format(sentence, e))
        statefulAggregatedSentence['reliability'] = 0.0

    return statefulAggregatedSentence


def processSentences(transcriptionLineDetailsFrame):

    # Several indices over the data were establshed to perform the
    # aggregation. hey are no longer required and a more informative index is
    # the index of the best matching line on the page.

    transcriptionLineDetailsReIndexed = transcriptionLineDetailsFrame.reset_index(
        level=[1, 2, 3, 4], drop=True)
    transcriptionLineDetailsReIndexed.set_index(
        'bestLineIndex', append=True, inplace=True)

    lineGroupedTranscriptionLineDetails = transcriptionLineDetailsReIndexed.groupby(
        level=[0, 1]).aggregate({
            "words":
            aggregateSentences,
            'subjectKey':
            lambda x: x.iloc[0],
            'y1':
            np.mean,
            'y2':
            np.mean,
            'x1':
            np.mean,
            'x2':
            np.mean,
            'transcriptionIndex':
            lambda x: tuple([xi for xi in x]),
            'numLines':
            lambda x: tuple([xi for xi in x])
        })

    lineGroupedTranscriptionLineDetails = lineGroupedTranscriptionLineDetails.reset_index(
        level=[1])

    lineGroupedTranscriptionLineDetails = pd.merge(
        lineGroupedTranscriptionLineDetails,
        subjectsFrame,
        left_index=True,
        right_index=True,
        how='left')

    return lineGroupedTranscriptionLineDetails

# ## Save the "most popular" transcriptions
# Also attempt to filter out double words e.g. `cheese cheese` and
# adjacent lines with a large fraction of shared text that are likely to
# erroneously repeated.


def saveAggregatedData(lineGroupedTranscriptionLineDetails, aggregatedDataCsvFileName, aggregatedDataSubjectWiseCsvFileName):

    aggregatedDataFile = open(aggregatedDataCsvFileName, 'w')
    aggregatedDataSubjectWiseFile = open(
        aggregatedDataSubjectWiseCsvFileName, 'w')
    # detailedAggregatedDataFile = open(aggregatedDataFileName, 'w')
    currentSubject = -1
    lastConsensusSentenceWords = (0, [])
    for index, row in lineGroupedTranscriptionLineDetails.iterrows():
        print("save aggregate row", row['words'])
        if index != currentSubject:
            if currentSubject != -1:
                aggregatedDataSubjectWiseFile.write('"\n')
            aggregatedDataSubjectWiseFile.write(
                '{0}@@{1}@@{2}@@"'.format(index, row['img_id'], row['url']))

            currentSubject = index

        # count the number of transcriptions that contributed to a sentence in case a deadlock between
        # duplicate lines must be broken - currently not used
        numTranscribedWords = functools.reduce(
            lambda total, increment: total + increment,
            [len(wordlist) for wordlist in row['words']['words']], 0)

        consensusSentenceWords = [
            wordlist[0].word for wordlist in row['words']['words']
            if len(wordlist) > 0
        ]
        consensusSentence = ' '.join(consensusSentenceWords)

        doubleWordRegex = r' ([^ ]{2,}) \1 ?'
        doubleWordMatch = regex.search(
            pattern=doubleWordRegex, string=consensusSentence)
        cleanConsensusSentence = consensusSentence
        if applyDoubleWordFilter and doubleWordMatch is not None:
            cleanConsensusSentence = regex.sub(
                pattern=doubleWordRegex,
                repl=lambda match: ' ' + match.group(1) + ' ',
                string=consensusSentence)
            if extraVerbose:
                print('Found double word {} in "{}" => {}'.format(
                    doubleWordMatch.group(1), consensusSentence,
                    cleanConsensusSentence))

        #print("cleanConsensusSentence", cleanConsensusSentence)
        cleanConsensusWords = cleanConsensusSentence.split(' ')

        # Identify duplicate sentences after double word removal (currently only
        # exact duplicates)
        lineWordIntersection = [
            word for word in cleanConsensusWords
            if word in lastConsensusSentenceWords[1]
        ]
        if applyDoubleLineFilter and (len(lineWordIntersection) == max(
                len(cleanConsensusWords), len(lastConsensusSentenceWords[1]))):
            if extraVerbose:
                print('Found duplicate sentence "{}" == "{}" ({})'.format(
                    cleanConsensusSentence, ' '.join(lastConsensusSentenceWords[
                        1]), lineWordIntersection))
            # Do not write the duplicate sentence to file
            continue

        # only update if the current sentence was not a duplicate of the
        # previous.
        lastConsensusSentenceWords = (
            numTranscribedWords, consensusSentenceWords)

        dataFileWriter = csv.writer(aggregatedDataFile)
        dataFileWriter.writerow(['{0}@@{1}@@{2}@@{3}@@{4}@@{5}'.format(
            currentSubject,
            row['img_id'],
            row['bestLineIndex'],
            '"' + cleanConsensusSentence + '"',
            #'(' + str(row['y1']) + ', ' + str(row['y2']),
            [str(len(wordlist)) for wordlist in row['words']['words']],
            # row['numLines'],
            row['url'])])
        # Note "<br />" line break sequence added at request of Huntington
        aggregatedDataSubjectWiseFile.write(
            '{0}<br />'.format(cleanConsensusSentence))
    aggregatedDataFile.close()
    aggregatedDataSubjectWiseFile.close()


if __name__ == '__main__':

    # Processing multiple classificaton files

    classificationCsvFiles = glob.glob(
        '{}/dcw*.csv'.format(classificationBaseDirectory))

    for sampleDataFileName in classificationCsvFiles:
        print('Processing {}...'.format(sampleDataFileName))
        # mssLabel = remainingClassificationCsvFiles[ledgerIndex].split('/')[-1][len(
        #     'classification_export_'):-4]
        mssLabel = sampleDataFileName.split('/')[-1][len(
            'classification_export_'):-4]
        print(mssLabel)
        # databaseName = databaseNamePattern.format(
        #     mss_label=mssLabel
        # )
        # 'dcwConsensusDoubleLineFix' if applyDoubleLineFix else 'dcwConsensus'
        # sampleDataFileName = classificationCsvFiles[
        #     ledgerIndex]  # 'decoding-the-civil-war-classifications-2.csv'
        aggregatedDataFileName = aggregatedDataFileNamePattern.format(
            mss_label=mssLabel)  # 'decoding-the-civil-war-aggregated.txt'
        aggregatedDataCsvFileName = aggregatedDataCsvFileNamePattern.format(
            mss_label=mssLabel)  # 'decoding-the-civil-war-consensus-linewise.csv'
        aggregatedDataSubjectWiseCsvFileName = aggregatedDataSubjectWiseCsvFileNamePattern.format(
            mss_label=mssLabel)  # 'decoding-the-civil-war-consensus-subjectwise.csv'
        identifiedLineFilePath = identifiedLineFilePathPattern.format(
            mss_label=mssLabel)  # 'dataWithLineIDs_subset.pkl'

        print(aggregatedDataFileName)
        print(aggregatedDataCsvFileName)
        print(aggregatedDataSubjectWiseCsvFileName)
        print(identifiedLineFilePath)

        subjectDataFileName = 'cce-transcription-test-subjects.csv'

        subjectsFrame = loadSubjectData(subjectDataFileName)

        titles, copyrights, publishdates, nTask1Parsed, nTask8Parsed = loadTelegrams(sampleDataFileName)
        print('Parsed {} titles and stored {}.'.format(
            nTask1Parsed, len(titles)))

        transcriptionLineStats, transcriptionLineDetailsFrame = processLoadedTelegrams(
            titles)

        #print("TranscriptionLineStats", transcriptionLineStats)
        #print("transcriptionLineDetailsFrame", transcriptionLineDetailsFrame)
        transcriptionLineDetailsFrame = groupTranscriptionsLinewise(
            transcriptionLineDetailsFrame, 3, identifiedLineFilePath, saveIdentifiedLineDetails)
        # This is an intentional no-op
        transcriptionLineDetailsFrame = doubleLineFix(
            transcriptionLineDetailsFrame, applyDoubleLineFix=False)

        # The previous step is time consuming so serialize the processed data at
        # this stage
        if saveIdentifiedLineDetails:
            transcriptionLineDetailsFrame.to_pickle(identifiedLineFilePath)

        lineGroupedTranscriptionLineDetails = processSentences(
            transcriptionLineDetailsFrame)
        saveAggregatedData(lineGroupedTranscriptionLineDetails,
                           aggregatedDataCsvFileName, aggregatedDataSubjectWiseCsvFileName)
