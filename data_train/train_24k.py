import cntk as C
import numpy as np
from data_provider import provider
from model_lf0 import SRU_MULTI_SPEAKER
import time
import datetime
import sys
import math

gpu_des = C.gpu(0)

C.device.try_set_default_device(gpu_des)

srate = '24k'

# define provider for two datasets
provider_train_N1 = provider('../data_preprocess/' + srate + '/list_N1', 87, 193, '../data_preprocess/' + srate + '/lab_N1/', '../data_preprocess/' + srate + '/cmp_N1/', 7500, 4, 'train', 1)
provider_train_N2 = provider('../data_preprocess/' + srate + '/list_N2', 87, 193, '../data_preprocess/' + srate + '/lab_N2/', '../data_preprocess/' + srate + '/cmp_N2/', 9000, 4, 'train', 2)
provider_train_N3 = provider('../data_preprocess/' + srate + '/list_N3', 87, 193, '../data_preprocess/' + srate + '/lab_N3/', '../data_preprocess/' + srate + '/cmp_N3/', 5500, 4, 'train', 3)
provider_test_N1 = provider('../data_preprocess/' + srate + '/list_N1', 87, 193, '../data_preprocess/' + srate + '/lab_N1/', '../data_preprocess/' + srate + '/cmp_N1/', 7500, 8, 'valid', 1)
provider_test_N2 = provider('../data_preprocess/' + srate + '/list_N2', 87, 193, '../data_preprocess/' + srate + '/lab_N2/', '../data_preprocess/' + srate + '/cmp_N2/', 9000, 8, 'valid', 2)
provider_test_N3 = provider('../data_preprocess/' + srate + '/list_N3', 87, 193, '../data_preprocess/' + srate + '/lab_N3/', '../data_preprocess/' + srate + '/cmp_N3/', 5500, 8, 'valid', 3)

# load model, trainer
lr = 0.0005
SRU = SRU_MULTI_SPEAKER(87, 193, lr, 0.5)
trainer = SRU.trainer

# settings
max_epoch = 200
cur_epoch = 0

# If restoration needed
#epoch_restore = 22
#SRU.trainer.restore_from_checkpoint('net/trainer_' + str(epoch_restore))
#cur_epoch += epoch_restore
#lr = 0.008
#SRU.learner.reset_learning_rate(C.learning_rate_schedule(lr, C.UnitType.sample))

# deal with validation loss increate
incr_count = 0 
pre_val_loss = sys.float_info.max

# training
while(cur_epoch < max_epoch):
    f = open('record_24k', 'a+')
    cur_epoch += 1

    train_total_loss = []
    test_total_loss_N1 = []
    test_total_loss_N2 = []
    test_total_loss_N3 = []

    provider_train_N1.reset()
    provider_train_N2.reset()    
    provider_train_N3.reset()    

    reset_count = 0

    start_time = time.time()
    print str(datetime.datetime.now()).split('.')[: -1][0]

    while reset_count < 3:

        if provider_train_N1.end_reading:
            provider_train_N1.reset()
            reset_count += 1

        if provider_train_N2.end_reading:
            provider_train_N2.reset()
            reset_count += 1
	
        if provider_train_N3.end_reading:
            provider_train_N3.reset()
            reset_count += 1

        lin_N1, aco_N1, idx_N1 = provider_train_N1.load_one_batch()
        lin_N2, aco_N2, idx_N2 = provider_train_N2.load_one_batch()
        lin_N3, aco_N3, idx_N3 = provider_train_N3.load_one_batch()

        trainer.train_minibatch({SRU.lin : lin_N1 + lin_N2 + lin_N3, SRU.aco : aco_N1 + aco_N2 + aco_N3, SRU.idx : np.concatenate((idx_N1, idx_N2, idx_N3))})

        train_total_loss.append(trainer.previous_minibatch_loss_average)

    provider_test_N1.reset()
    provider_test_N2.reset()
    provider_test_N3.reset()

    while not provider_test_N1.end_reading:

       lin_N1, aco_N1, idx_N1 = provider_test_N1.load_one_batch()

       test_total_loss_N1.append(trainer.test_minibatch({SRU.lin : lin_N1, SRU.aco : aco_N1, SRU.idx : idx_N1})) 
    
    while not provider_test_N2.end_reading:

        lin_N2, aco_N2, idx_N2 = provider_test_N2.load_one_batch()

        test_total_loss_N2.append(trainer.test_minibatch({SRU.lin : lin_N2, SRU.aco : aco_N2, SRU.idx : idx_N2}))
    
    
    while not provider_test_N3.end_reading:

        lin_N3, aco_N3, idx_N3 = provider_test_N3.load_one_batch()

        test_total_loss_N3.append(trainer.test_minibatch({SRU.lin : lin_N3, SRU.aco : aco_N3, SRU.idx : idx_N3}))

    train_ave_loss = np.mean(np.asarray(train_total_loss))
    test_ave_loss_N1 = np.mean(np.asarray(test_total_loss_N1))
    test_ave_loss_N2 = np.mean(np.asarray(test_total_loss_N2))
    test_ave_loss_N3 = np.mean(np.asarray(test_total_loss_N3))

    end_time = time.time()

    print('epoch : %i, train_ave_loss : %f, test_ave_loss_N1 : %f, test_ave_loss_N2 : %f, test_ave_loss_N3 : %f, duration : %f' % (cur_epoch, train_ave_loss, test_ave_loss_N1, test_ave_loss_N2, test_ave_loss_N3, (end_time - start_time)))

    info = str(cur_epoch) + ', ' + str(train_ave_loss) + ', ' + str(test_ave_loss_N1) + ', ' + str(test_ave_loss_N2) + ', ' + str(test_ave_loss_N3)   

    SRU.trainer.save_checkpoint('net/' + srate + '/trainer_' + str(cur_epoch))

    if (test_ave_loss_N1 + test_ave_loss_N2 + test_ave_loss_N3) / 2.0 < pre_val_loss and not math.isnan(test_ave_loss_N3):
        incr_count = 0
        pre_val_loss = (test_ave_loss_N1 + test_ave_loss_N2 + test_ave_loss_N3) / 2.0
        #lr = 0.004
        #SRU.learner.reset_learning_rate(C.learning_rate_schedule(lr, C.UnitType.sample))
    else:
        incr_count += 1
        SRU.trainer.restore_from_checkpoint('net/' + srate + '/trainer_' + str(cur_epoch - incr_count))
        #lr = lr * 0.5
        #SRU.learner.reset_learning_rate(C.learning_rate_schedule(lr, C.UnitType.sample))
        print 'trainer restore to ' + str(cur_epoch - incr_count) + ', lr reduced'
        info += ', restored to ' + str(cur_epoch - incr_count)
    f.writelines(info + '\n')
    f.close()

