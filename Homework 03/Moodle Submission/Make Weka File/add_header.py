import csv

rows = list()
for i in range(1, 17):
    rows.append("A{x}".format(x=i))

with open('..\\data\\sample.csv', 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(rows)

    with open('..\\data\\sample.txt', 'r') as incsv:
        reader = csv.reader(incsv)
        writer.writerows(row for row in reader)
