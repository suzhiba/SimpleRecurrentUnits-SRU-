import numpy as np
import random
from io_funcs.binary_io import BinaryIOCollection


class provider(object):

    def __init__(self, list_path, dim_lab, dim_cmp, root_lab, root_cmp, train_num, batch_size, mode, index):

        f_list = open(list_path, 'r')
        lines_list = f_list.readlines()
        f_list.close()

        self.dim_lab = dim_lab
        self.dim_cmp = dim_cmp
        self.list_labels = [root_lab + item.split()[0] + '.lab' for item in lines_list]
        self.list_cmp = [root_cmp + item.split()[0] + '.cmp' for item in lines_list]

        for i in range(0, len(self.list_labels)):
            assert self.list_labels[i].split('.')[0].split('/')[-1] == self.list_cmp[i].split('.')[0].split('/')[-1]

        self.list_index = 0
        self.end_reading = False
        self.io_tool = BinaryIOCollection()
        self.batch_size = batch_size

        if mode == 'train':
            self.list_labels_using = self.list_labels[: train_num]
            self.list_cmp_using = self.list_cmp[: train_num]
            self.len_list = len(self.list_labels_using)
        if mode == 'valid':
            self.list_labels_using = self.list_labels[train_num :]
            self.list_cmp_using = self.list_cmp[train_num :]
            self.len_list = len(self.list_labels_using)

        if index == 1:
            self.index_array = np.asarray([1, 0, 0])
            self.index_array = np.tile(self.index_array.reshape(1, -1), (self.batch_size, 1)).astype(np.float32)
        elif index == 2:
            self.index_array = np.asarray([0, 1, 0])
            self.index_array = np.tile(self.index_array.reshape(1, -1), (self.batch_size, 1)).astype(np.float32)
	elif index == 3:
            self.index_array = np.asarray([0, 0, 1])
            self.index_array = np.tile(self.index_array.reshape(1, -1), (self.batch_size, 1)).astype(np.float32)

        else:
            raise Exception('out of index')

    def reset(self):

        self.list_index = 0
        self.end_reading = False
        c = list(zip(self.list_labels_using, self.list_cmp_using))
        random.shuffle(c)
        self.list_labels_using, self.list_cmp_using = zip(*c)
        self.list_labels_using = list(self.list_labels_using)
        self.list_cmp_using = list(self.list_cmp_using)

    def load_one_batch(self):

        list_input = []
        list_target = []
     
        for i in range(0, self.batch_size):

            #print self.list_labels_using[self.list_index]
            labs = self.io_tool.load_binary_file(self.list_labels_using[self.list_index], self.dim_lab)
            cmps = self.io_tool.load_binary_file(self.list_cmp_using[self.list_index], self.dim_cmp)
            

            assert labs.shape[0] == cmps.shape[0]

            list_input.append(labs.astype(np.float32))
            list_target.append(cmps.astype(np.float32))            

            self.list_index += 1

            if self.list_index + self.batch_size - 1 >= self.len_list:

                self.end_reading = True

        return list_input, list_target, self.index_array


