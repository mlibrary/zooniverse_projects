import csv
import requests

with open('s3photos.csv') as csvfile:
    csvrows = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in csvrows:
        filename = row[0]
        url = row[1]
        print(url)
        result = requests.get(url, stream=True)
        if result.status_code == 200:
            image = result.raw.read()
            open(filename,"wb").write(image)

            
            #code from https://www.quora.com/How-do-I-write-a-Python-code-to-download-images-from-100-URLs-stored-in-a-CSV-file
