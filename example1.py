import SeqGAN_PyTorch
from SeqGAN_PyTorch import ACSeqGAN
import torch

model = ACSeqGAN('toy', 'mol_metrics', params={'PRETRAIN_GEN_EPOCHS':200,
                                               'PRETRAIN_DIS_EPOCHS':2, 
                                               'd_num_classes': 1})
model.load_training_set('./qm9_5k.csv')
# model.set_training_program(['novelty'], [1])
model.set_training_program(['druglikeliness'], [100])
model.load_metrics()
# model.load_prev_pretraining(ckpt='ckpt/toy_pretrain_ckpt.pth')
model.train(ckpt_dir='ckpt')


# import SeqGAN_PyTorch
# from SeqGAN_PyTorch import ACSeqGAN
# import torch
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# # Initialize the model
# model = ACSeqGAN('toy', 'mol_metrics', params={'PRETRAIN_GEN_EPOCHS': 100,
#                                                'PRETRAIN_DIS_EPOCHS': 2,
#                                                'd_num_classes': 1})

# model.load_training_set('./qm9_5k.csv')
# model.set_training_program(['druglikeliness'], [100])
# model.load_metrics()

# # Define a custom training function
# def train_fn(model):
#     model.train(ckpt_dir='ckpt')

# # Compile the training function
# compiled_train_fn = torch.compile(train_fn)

# # Run the compiled training function
# compiled_train_fn(model)
