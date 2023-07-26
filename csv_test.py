import csv 
spamReader = csv.reader(open('C:/Users/MasterPC/pythonprogs/LUNGS/readme/DE.csv', newline=''), delimiter=',', quotechar='|')
for row in spamReader:
	print(row[0]+" +++ "+row[2])