import os

f = open('bad.txt')
lines = f.readlines()
for line in lines:
    index = line.split()[0]
    os.remove('16k_all/lab_N3/' + index + '.lab')
    os.remove('16k/lab_N3/' + index + '.lab')
    os.remove('16k_all/cmp_N3/' + index + '.cmp')
    os.remove('16k/cmp_N3/' + index + '.cmp')
