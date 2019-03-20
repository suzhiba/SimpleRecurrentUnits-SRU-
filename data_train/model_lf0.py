import cntk as C

class SRU_MULTI_SPEAKER(object):

    def __init__(self, n_in, n_out, init_lr, init_momentum):

        self.dim_in = int(n_in)
        self.dim_out = int(n_out)
        self.init_lr = init_lr
        self.init_momentum = init_momentum

        self.lin = C.sequence.input_variable(shape=(self.dim_in), name='input')
        self.aco = C.sequence.input_variable(shape=(self.dim_out), name='output')
        self.idx = C.input_variable(shape=(3), name='index')

        self.create_parameters()
        self.output = self.model(self.lin, self.idx)

        #self.loss = self.gen_loss(self.output, self.aco)
        self.loss = self.gen_loss2(self.output, self.aco)
        self.loss3 = self.gen_loss3(self.output, self.aco)
   
        self.lr_s = C.learning_rate_schedule(self.init_lr, C.UnitType.sample)
        self.mom_s = C.momentum_schedule(self.init_momentum)
        self.learner = C.momentum_sgd(self.output.parameters, lr=self.lr_s, momentum=self.mom_s, gradient_clipping_threshold_per_sample=1.0)
        #self.learner = C.adam(self.output.parameters, lr=self.lr_s, momentum=self.mom_s)
        self.trainer = C.Trainer(self.output, (self.loss, self.loss), [self.learner])
       
    def create_bias(self, number):

        self.list_bias = []
        for i in xrange(number):
            self.list_bias.append(C.parameter(shape=(512), name='bias_' + str(i)))


    def create_parameters(self):

        self.dnn_three = C.layers.Sequential([
                             C.layers.Dense(1024, activation=C.tanh, name='dnn_head_1'),
                             C.layers.Dense(1024, activation=C.tanh, name='dnn_head_2'),
                             C.layers.Dense(1024, activation=C.tanh, name='dnn_head_3')])

        self.dnn_1 = C.layers.Dense(8 * 512, bias=False, name='dnn_sru_1')
        self.dnn_2 = C.layers.Dense(8 * 512, bias=False, name='dnn_sru_2')
        self.dnn_3 = C.layers.Dense(8 * 512, bias=False, name='dnn_sru_3')
        self.dnn_4 = C.layers.Dense(8 * 512, bias=False, name='dnn_sru_4')

        self.dnn_final = C.layers.Dense(self.dim_out, name='dnn_final')

        self.create_bias(16)

        self.emd = C.layers.Embedding(shape=(16), init=C.initializer.uniform(0.1), name='emd')

        self.dnn_emd_input = C.layers.Dense(16, activation=C.tanh, name='dnn_emd_input')

        self.dnn_emd_rec_for = C.layers.Dense(512, activation=C.tanh, name='dnn_emd_rec_for')
        self.dnn_emd_rec_bac = C.layers.Dense(512, activation=C.tanh, name='dnn_emd_rec_bac')

    def bsru_layer(self, sru_1, index, init_for, init_bac):

        f_1_f = C.sigmoid(sru_1[0 * 512 : 1 * 512] + self.list_bias[0 + index * 4])
        r_1_f = C.sigmoid(sru_1[1 * 512 : 2 * 512] + self.list_bias[1 + index * 4])
        c_1_f_r = (1 - f_1_f) * sru_1[2 * 512 : 3 * 512]

        dec_c_1_f = C.layers.ForwardDeclaration('f_' + str(index))
        var_c_1_f = C.sequence.delay(dec_c_1_f, initial_state=init_for, time_step=1)
        nex_c_1_f = var_c_1_f * f_1_f + c_1_f_r
        dec_c_1_f.resolve_to(nex_c_1_f)

        h_1_f = r_1_f * C.tanh(nex_c_1_f) + (1 - r_1_f) * sru_1[3 * 512 : 4 * 512]

        f_1_b = C.sigmoid(sru_1[4 * 512 : 5 * 512] + self.list_bias[2 + index * 4])
        r_1_b = C.sigmoid(sru_1[5 * 512 : 6 * 512] + self.list_bias[3 + index * 4])
        c_1_b_r = (1 - f_1_b) * sru_1[6 * 512 : 7 * 512]

        dec_c_1_b = C.layers.ForwardDeclaration('b_' + str(index))
        var_c_1_b = C.sequence.delay(dec_c_1_b, initial_state=init_bac, time_step=-1)
        nex_c_1_b = var_c_1_b * f_1_b + c_1_b_r
        dec_c_1_b.resolve_to(nex_c_1_b)

        h_1_b = r_1_b * C.tanh(nex_c_1_b) + (1 - r_1_b) * sru_1[7 * 512 : 8 * 512]

        x = C.splice(h_1_f, h_1_b)

        return x

    def model(self, lin, idx):
        
        emd = self.emd(idx)
        x = C.splice(lin, C.sequence.broadcast_as(self.dnn_emd_input(emd), lin))
        x = self.dnn_three(x)

        init_for = C.sequence.broadcast_as(self.dnn_emd_rec_for(emd), lin)
        init_bac = C.sequence.broadcast_as(self.dnn_emd_rec_bac(emd), lin)

        sru_1 = self.dnn_1(x)
        x = self.bsru_layer(sru_1, 0, init_for, init_bac)
        
        sru_1 = self.dnn_2(x)
        x = self.bsru_layer(sru_1, 1, init_for, init_bac)

        sru_1 = self.dnn_3(x)
        x = self.bsru_layer(sru_1, 2, init_for, init_bac)

        sru_1 = self.dnn_4(x)
        x = self.bsru_layer(sru_1, 3, init_for, init_bac)

        return self.dnn_final(x)     

    def gen_loss(self, output, aco):

        L = C.sequence.reduce_sum(C.reduce_sum(output * 0) + 1)
        return C.sequence.reduce_sum(C.reduce_sum(C.square(output - aco))) / L
        
    
    def gen_loss2(self, output, aco):

        L = C.sequence.reduce_sum(C.reduce_sum(output * 0) + 1)
        #return C.sequence.reduce_sum(C.square(output - aco)) / L
        return C.sequence.reduce_sum(C.reduce_sum(C.square(output - aco)) +10 * C.reduce_sum(C.square(output - aco)[180:183])) / L
        

    def gen_loss3(self, output, aco):

        L = C.sequence.reduce_sum(C.reduce_sum(output * 0) + 1)
        return C.sequence.reduce_sum(C.square(output - aco)) / L
        #return C.sequence.reduce_sum(C.reduce_sum(C.square(output - aco)) +10 * C.reduce_sum(C.square(output - aco)[180:183])) / L
