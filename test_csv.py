import csv
def give_arguments(arg):
	csv_rows=[]
	spamReader = csv.reader(open('C:/Users/MasterPC/Desktop/running_keras_info.csv'), delimiter=';', quotechar='|')
	for row in spamReader:
		csv_rows.append(row)


	return csv_rows[arg]

#print(csv_rows[ddt][2])