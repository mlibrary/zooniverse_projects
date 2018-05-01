# Introduction
This document summarizes steps needed in order to get consensus results on the CCE Transcription project.

## How to get consensus results on the CCE classification

1) Download or clone the script file.
Please refer https://github.com/hughdickinson/DCWConsensus to see the logic on measuring the consensus.

2) Export classification and subject data interested from Zooniverse

3) Open the script called ‘DcwAggregation.py” and set the corresponding classification directory, subject data directory, workflow id, and date as below.

```
classificationBaseDirectory
workflow_id
liveDate
subjectDataFileName = 'cce-transcription-test-subjects.csv'
```

4) loadTelegrams function in the script parse annotated data by a task. Set appropriate task ids and create line collecting data structures if necessary. For example, T1 is about title annotation and below part is added to parse title annotations.

```
def loadTelegrams():
...
titlesLines = TelegramLines()
…
elif task['task'] == "T1" and len(task['value']) > 0:
   isTask1Exist = True
   # process the lines that were transcribed for this task
   for taskValueItem in task['value']:
       titleLine = TextLine(
           taskValueItem['x1'], taskValueItem['y1'],
           taskValueItem['x2'], taskValueItem['y2'],
           taskValueItem['details'][0]['value'])
       titlesLines.addLine(titleLine)
...
```

5) Check which subject/task to be processed by passing the right parameter to the processLoadedTelegrams function. For example, below only considers measuring consensus on titles.

```
transcriptionLineStats, transcriptionLineDetailsFrame = processLoadedTelegrams(
   titles)
```

6) Run the script
```
$ python DcwAggregation.py
```

7) Check the consensus results file currently named as decoding-the-civil-war-consensus-linewise_-classifications and decoding-the-civil-war-consensus-subjectwise_-classifications_withBreaks


## Possible future works for generalizing
There are some possible future works for improving this script. First, setting the tasks and parsing the corresponding annotations can be more generalized by providing a meta file such as xml that can set information about tasks ids, tasks types, and their line tolerances from a project owner, and then iterating the data parsing(loadTelegrams() according to the metadata file. Secondly, instead of hard coded directory path and work flow id, these can be also provided in a separate file, giving a project owner more flexibility in managing these information.
