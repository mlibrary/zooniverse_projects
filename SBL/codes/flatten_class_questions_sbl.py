import csv
import json
import sys

csv.field_size_limit(sys.maxsize)
# file location area:
location = r'/Users/stellachoi/Documents/SDL/zooniverse/SBL/' \
           r'south-bend-lead-identification-project-classifications-csv.csv'
out_location = r'/Users/stellachoi/Documents/SDL/zooniverse/SBL/flatten_classification_south-bend-lead.csv'
#name_location = r'C:\py\FFIPusers\IPuser.csv'

# define functions area:


def include(class_record):
    if int(class_record['workflow_id']) == 6071:
        pass
    else:
        return False
    if float(class_record['workflow_version']) >= 41.75:
        pass
    else:
        return False
    return True


with open(out_location, 'w', newline='') as file:
    # Note we have added a number of fields including 'line_number and changed the order to suit our whims -
    # The write statement must match both items and order
    fieldnames = ['line_number',
                  'subject_ids',
                  'user_name',
                  'workflow_id',
                  'workflow_version',
                  'classification_id',
                  'created_at',
                  'testresult_exist',
                  'testresult_list',
                  'testresult_classify']

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # Initialize and load pick lists
    i = 0
    j = 0
    #name_list = load_pick_ip()
    task_answer_template_1 = ['Yes', 'No']
    task_answer_template_3 = ['Map', 'Letter/Communication', 'City Directory', 'Photograph', 'Graph',
                              'Signed Form', 'Receipt/Invoice/Financial Statement', 'Report (Government or Private)',
                              'Other/Unknown']
    with open(location) as f:
        r = csv.DictReader(f)
        for row in r:
            i += 1
            if i == 3000:
                break
            if include(row) is True:
                j += 1
                metadata = json.loads(row['metadata'])
                annotations = json.loads(row['annotations'])

                # Area to add the blocks that work on each record of the classification

                # generate user_name for not_signed_in users
                user_name = str(row['user_name'])
                if row['user_id'] == '':
                    user_name = 'Visitor'
                    #  user_name = str(row[user_name])
                    #  user_name = row['user_ip']

                task_vector_1 = [0, 0]
                task_answer_2 = ''
                task_vector_3 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                for task in annotations:
                    # Even though the first question only allows single answers we will use Block 4 - Multiple-
                    # allowed answer question - answer vector method to record it:
                    try:
                        if 'T2' == task['task']:
                            for k in range(0, 2):
                                if task_answer_template_1[k] in task['value']:
                                    task_vector_1[k] = 1
                            if task_vector_1 == [0, 0]:  # no answers selected
                                task_vector_1 = []
                    except (TypeError, KeyError):
                        task_vector_1 = ''
                        continue

                    # The second question is a single required answer with short answers that need no massaging
                    try:
                        if 'T1' == task['task']:
                            if task['value'] is not None:
                                task_answer_2 = str(task['value'])
                    except KeyError:
                        continue
                    # The third question
                    try:
                        if 'T3' == task['task']:
                            for task_value in task['value']:
                                if task_value in task_answer_template_3:
                                    k = task_answer_template_3.index(task_value)
                                    task_vector_3[k] = 1
                    except (TypeError, KeyError):
                        continue

                # Writer must agree with open field names and assign correct values to the fields
                writer.writerow({'line_number': str(i),
                                 'subject_ids': row['subject_ids'],
                                 'user_name': user_name,
                                 'workflow_id': row['workflow_id'],
                                 'workflow_version': row['workflow_version'],
                                 'classification_id': row['classification_id'],
                                 'created_at': row['created_at'],
                                 'testresult_exist': task_vector_1,
                                 'testresult_list': task_answer_2,
                                 'testresult_classify': task_vector_3})
            print(i, j)
        # Area to print final status report
        print(i, 'lines read and inspected', j, 'records processed and copied')

