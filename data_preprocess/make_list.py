import os

dirname = '48k_all'

file_names = os.listdir(dirname + '/lab_N1')
cmp_names = os.listdir(dirname + '/cmp_N1')

file_names_new = []
for file_name in file_names:
    if file_name[:-3] + 'cmp' in cmp_names:
        file_names_new.append(file_name)
 
f = open(dirname + '/list_N1', 'w')

file_ids = [item.split()[0].split('.')[0] + '\n' for item in file_names_new]

f.writelines(file_ids)

f.close()


file_names = os.listdir(dirname + '/lab_N2')
cmp_names = os.listdir(dirname + '/cmp_N2')

file_names_new = []
for file_name in file_names:
    if file_name[:-3] + 'cmp' in cmp_names:
        file_names_new.append(file_name)
 
f = open(dirname + '/list_N2', 'w')

file_ids = [item.split()[0].split('.')[0] + '\n' for item in file_names_new]

f.writelines(file_ids)

f.close()


file_names = os.listdir(dirname + '/lab_N3')
cmp_names = os.listdir(dirname + '/cmp_N3')

file_names_new = []
for file_name in file_names:
    if file_name[:-3] + 'cmp' in cmp_names:
        file_names_new.append(file_name)

f = open(dirname + '/list_N3', 'w')

file_ids = [item.split()[0].split('.')[0] + '\n' for item in file_names_new]

f.writelines(file_ids)

f.close()

