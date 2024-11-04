import SeqGAN_PyTorch
from SeqGAN_PyTorch import ACSeqGAN

model = ACSeqGAN('test', 'mol_metrics', params={'PRETRAIN_DIS_EPOCHS': 2, 
                                                'd_num_classes': 2,
                                                'GENERATED_NUM': 500,
                                                'PRETRAIN_GEN_EPOCHS':100})
model.load_training_set('./train_small.csv')
model.set_training_program(['novelty'], [1])
model.load_metrics()
# model.pretrain()
model.load_prev_pretraining(ckpt='checkpoints/test_pretrain_ckpt.pth')
model.train(ckpt_dir='checkpoints/')
