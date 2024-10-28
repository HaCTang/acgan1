import SeqGAN_PyTorch
from SeqGAN_PyTorch import ACSeqGAN

model = ACSeqGAN('test', 'mol_metrics', params={'PRETRAIN_DIS_EPOCHS': 1, 
                                                'd_num_classes': 2})
model.load_training_set('./train_small.csv')
model.set_training_program(['novelty'], [1])
model.load_metrics()
model.pretrain()
