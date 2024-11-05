import SeqGAN_PyTorch
from SeqGAN_PyTorch import ACSeqGAN

ACSeqGAN_params = {
    'PRETRAIN_GEN_EPOCHS': 250, 'PRETRAIN_DIS_EPOCHS': 10, 'MAX_LENGTH': 60, 'LAMBDA_1': 0.5, "DIS_EPOCHS": 2, 'GENERATED_NUM': 6400}

# hyper-optimized parameters
disc_params = {"d_l2reg": 0.2, "d_emb_dim": 32, 'd_num_classes': 1, "d_filter_sizes": [
    1, 2, 3, 4, 5, 8, 10, 15], "d_num_filters": [50, 50, 50, 50, 50, 50, 50, 75], "d_dropout": 0.75}

ACSeqGAN_params.update(disc_params)

model = ACSeqGAN('qm9-5k', 'mol_metrics', params=ACSeqGAN_params)
model.load_training_set('./qm9_5k.csv')
# model.load_prev_pretraining('pretrain_ckpt/qm9-5k_pretrain_ckpt')
model.set_training_program(
    ['druglikeliness'], [100])
model.load_metrics()
# model.load_prev_training(ckpt='qm9-5k_20.ckpt')
model.train()
