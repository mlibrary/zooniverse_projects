# Introduction
This document summarizes steps needed in order to get consensus results on Unearthing Michigan Ecological Data.
https://www.zooniverse.org/projects/jmschell/unearthing-michigan-ecological-data/

## How to get consensus results on the classification

1) Flatten the classification data. Properly set the location(exported classification file) and out_location in the code(flatten_class_questions_ume.py) as well as tasks types.

```
$ python flatten_class_questions_ume.py
```

2) Sort the flattened data by subject ids. Properly set the input(output file in the step1) and output file in the code.

```
$ python sort_flatten_class_ume.py
```

3) Aggregate the classification data by subject ids. Properly set the location and out location directory and files in the code as well as output field names and aggregation logics.
For example, for this project, we aggregate data on the answers of Task 1 asking types of data. Users can choose multiple types.

Output Columns in ‘aggregate_ume.csv’
* classifications - Number of classifications for this subject
* testresult_T1 - A vector counts on the Task 1(‘What kind of data is on this page?’). E.g [2, 0, 0, 0, 0, 0, 2, 0]
* testresult_T1_percent -  Percent on the each data type respect to the total number of people who answered.

```
$python aggregate_frame_ume.py
```
