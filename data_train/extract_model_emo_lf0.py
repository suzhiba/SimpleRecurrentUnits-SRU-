import cntk as C
import numpy as np
from io_funcs.binary_io import BinaryIOCollection
from model_lf0_weight import SRU_MULTI_SPEAKER

gpu_descriptor = C.gpu(3)

C.try_set_default_device(gpu_descriptor)

proj = SRU_MULTI_SPEAKER(87, 187, 0.001, 0.5)

trainer = proj.trainer

trainer.restore_from_checkpoint('net/16k/trainer_' + str(41))

output = trainer.model

index = C.Constant(value=np.asarray([0, 1, 0]).astype(np.float32))
input = C.sequence.input_variable(shape=87)

out = output(input, index)

out.save('extracted_model/16k/model_emo')
