from random import shuffle

srate = '48k_all'

f = open(srate + '/list_N1', 'r')

items = f.readlines()

f.close()

shuffle(items)

g = open(srate + '/list_N1', 'w')

g.writelines(items)

g.close()


f = open(srate + '/list_N2', 'r')

items = f.readlines()

f.close()

shuffle(items)

g = open(srate + '/list_N2', 'w')

g.writelines(items)

g.close()


f = open(srate + '/list_N3', 'r')

items = f.readlines()

f.close()

shuffle(items)

g = open(srate + '/list_N3', 'w')

g.writelines(items)

g.close()

