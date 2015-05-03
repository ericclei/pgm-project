import random, math

num_genes = 12331
num_diseases = 3215

def convert2num(row, col):
	return row * num_genes + col

def convert2coord(num):
	return num / num_genes, num % num_genes

done = set()

fin = open("Data/IMC/data.csv","r")
fout = open("Data/IMC/data_processed.csv","w")

for line in fin:
	fout.write(line)
	tokens = line.strip().split(",")
	done.add(convert2num(tokens[0], tokens[1]))

fin.close()

num_one = len(done)

while len(done) < 50 * num_one:
	num = random.randint(0, num_genes * num_diseases - 1)
	if num in done:
		continue
	done.add(num)
	coord = convert2coord(num)
	fout.write("{0},{1},0\n".format(coord[0], coord[1]))

fout.close()

# print 50 * num_one
# print math.floor(50 * num_one / 4)

test_indices = sorted(random.sample(xrange(50 * num_one), 10 * num_one))
index_test = 0

fin = open("Data/IMC/data_processed.csv","r")
fout_train = open("Data/IMC/data_train.csv", "w")
fout_test = open("Data/IMC/data_test.csv", "w")

for index, line in enumerate(fin):
	if index_test < len(test_indices) and index == test_indices[index_test]:
		fout_test.write(line)
		index_test += 1
	else:
		fout_train.write(line)
fout_train.close()
fout_test.close()

