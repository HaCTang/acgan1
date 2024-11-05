import SeqGAN_PyTorch
from SeqGAN_PyTorch import ACSeqGAN

model = ACSeqGAN('toy', 'mol_metrics', params={'PRETRAIN_DIS_EPOCHS': 1})
model.load_training_set('./toy.csv')
# model.set_training_program(['novelty'], [1])
model.set_training_program(['druglikeliness'], [100])
model.load_metrics()
model.train(ckpt_dir='ckpt')
