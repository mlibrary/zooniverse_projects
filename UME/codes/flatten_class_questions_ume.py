import csv
import json
import sys

csv.field_size_limit(sys.maxsize)
# file location area:
location = r'/Users/stellachoi/Documents/SDL/zooniverse/unearthing/' \
           r'official-release-workflow-classifications.csv'
out_location = r'/Users/stellachoi/Documents/SDL/zooniverse/unearthing/flatten_classification_ume.csv'
#name_location = r'C:\py\FFIPusers\IPuser.csv'

# define functions area:


def include(class_record):
    """
    if int(class_record['workflow_id']) == 8389:
        pass
    else:
        return False
    if float(class_record['workflow_version']) >= 1.1:
        pass
    else:
        return False
    """
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
                  'testresult_T1']

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # Initialize and load pick lists
    i = 0
    j = 0
    #name_list = load_pick_ip()
    task_answer_template_1 = ['Map', 'Graph', 'Diagram/Illustration', 'Species, listed or named',
                              'Table of numbers', 'Photograph', 'There is no data on this page. ',
                              'Other']

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

                task_vector_1 = [0, 0, 0, 0, 0, 0, 0, 0]

                for task in annotations:
                    try:
                        if 'T1' == task['task']:
                            for task_value in task['value']:
                                if task_value in task_answer_template_1:
                                    k = task_answer_template_1.index(task_value)
                                    task_vector_1[k] = 1
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
                                 'testresult_T1': task_vector_1})
            print(i, j)
        # Area to print final status report
        print(i, 'lines read and inspected', j, 'records processed and copied')

