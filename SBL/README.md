# Introduction
This document summarizes steps needed in order to get consensus results on the South Bend Lead project.

## How to get consensus results on the SBL classification

1) Flatten the classification data. Properly set the location(exported classification file) and out_location in the code(flatten_class_questions_sbl.py) as well as tasks types.

```
$ python flatten_class_questions_sbl.py
```

2) Sort the flattened data by subject ids. Properly set the input(output file in the step1) and output file in the code.

```
$ python sort_flatten_class_sbl.py
```

3) Aggregate the classification data by subject ids. Properly set the location and out location directory and files in the code as well as output field names and aggregation logics.
For example, for South Bend Lead project, we aggregate data on the answers of Task 2 asking if a sample is a test result documents or not and Task 3 asking listing of toxins or chemicals that user see in the sample. Then, those aggregations by subject ids were recorded in the output file(aggegate_south-bend-lead.csv).

Output Columns in ‘aggregate_south-bend-lead.csv’
* classifications - Number of classifications for this subject
* testresult_exist - A vector counts on Yes or No of the Task 2(‘Are there test results in this image?’). E.g [2, 0], 2 people said yes that this is a test result, no people said this is not a test.
* testresult_exist_yes_percent - Percentage of answering Yes on Task 2
* testresult_list - List of all toxins/chemicals people answered. Each answer is concatenated with ‘;’
* testresult_classify - A vector counts on document types that people classified. Options are ['Map', 'Letter/Communication', 'City Directory', 'Photograph', 'Graph', ‘Signed Form', 'Receipt/Invoice/Financial Statement', 'Report (Government or Private)', 'Other/Unknown'] in order. E.g [0, 0, 0, 4, 0, 0, 0, 1, 0] represents 4 people answered this document is Photograph, and 1 people answered Report (Government or Private).
* testresult_classify_percent - Percent on the each document type respect to the total answer. E.g [0.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 20.0, 0.0]

```
$python aggregate_frame_sbl.py
```
