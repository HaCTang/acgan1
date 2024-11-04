import SeqGAN_PyTorch
from SeqGAN_PyTorch import ACSeqGAN

model = ACSeqGAN('test', 'mol_metrics', params={'d_num_classes': 2,
                                                'MAX_LENGTH': 100,
                                                'GENERATED_NUM': 500,
                                                'PRETRAIN_GEN_EPOCHS':100,
                                                'PRETRAIN_DIS_EPOCHS':2})
# model.load_training_set('./train_NAPro.csv')
model.load_training_set('./train_small.csv')
model.set_training_program(['druglikeliness'], [100])
model.load_metrics()
# model.pretrain()
# model.load_prev_pretraining(ckpt='checkpoints/test_pretrain_ckpt.pth')
model.train(ckpt_dir='checkpoints/')
