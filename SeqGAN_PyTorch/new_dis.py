import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.nn.utils as utils

import numpy as np

class Discriminator(nn.Module):
    def __init__(self, sequence_length, num_classes, vocab_size, emb_dim, filter_sizes, use_cuda,
                 num_filters, dropout=0.2, l2_reg_lambda=1.0, wgan_reg_lambda=1.0, grad_clip=1.0):
        super(Discriminator, self).__init__()

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.filter_sizes = filter_sizes
        self.use_cuda = use_cuda
        self.num_filters = num_filters
        self.dropout = dropout
        self.l2_reg_lambda = l2_reg_lambda
        self.wgan_reg_lambda = wgan_reg_lambda
        self.grad_clip = grad_clip

        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filter, (filter_size, emb_dim))
            for filter_size, num_filter in zip(filter_sizes, num_filters)
        ])

        self.highway_transform = nn.Linear(sum(num_filters), sum(num_filters))
        self.highway_gate = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout_layer = nn.Dropout(p=dropout)

        self.lin_real_fake = nn.Linear(sum(num_filters), 1)
        self.classifier = nn.Linear(sum(num_filters), num_classes)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00005)
        self.auxiliary_loss = F.cross_entropy

        if self.use_cuda:
            self.to(torch.device("cuda"))

    def forward(self, x):
        if self.use_cuda:
            x = x.to(torch.device("cuda"))
        
        emb = self.emb(x).unsqueeze(1)
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(emb)).squeeze(3)
            pooled = F.max_pool2d(conv_out, (conv_out.size(2)))
            pooled_outputs.append(pooled.squeeze(2))

        h_pool = torch.cat(pooled_outputs, 1)
        print(h_pool.size())
        h_pool_flat = h_pool.flatten(start_dim=1)

        print(h_pool_flat.size())

        transform = F.relu(self.highway_transform(h_pool_flat))
        gate = torch.sigmoid(self.highway_gate(h_pool_flat))
        highway = gate * transform + (1.0 - gate) * h_pool_flat

        pred = self.dropout_layer(highway)
        real_fake_output = self.lin_real_fake(pred)
        class_output = self.classifier(pred)
        return real_fake_output, class_output

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
        interpolated = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolated, _ = self.forward(interpolated)
        fake = torch.ones(d_interpolated.size()).to(real_samples.device)
        gradients = grad(outputs=d_interpolated, inputs=interpolated,
                         grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def compute_loss(self, real_samples, fake_samples, real_class_pred, class_target):
        real_fake_loss = -torch.mean(real_samples) + torch.mean(fake_samples) # Wasserstein distance
        class_loss = self.auxiliary_loss(real_class_pred, class_target)
        gp = self.compute_gradient_penalty(real_samples, fake_samples)
        params = torch.cat([param.view(-1) for param in self.parameters()])
        l2_loss = torch.norm(params, p=2)
        l2_loss = self.l2_reg_lambda * l2_loss
        total_loss = real_fake_loss + self.wgan_reg_lambda * gp + class_loss + l2_loss
        return total_loss

    def train_step(self, x, real_labels, class_labels, fake_samples, dropout=0.2):
        if self.use_cuda:
            x = x.to(torch.device("cuda"))
            real_labels = real_labels.to(torch.device("cuda"))
            class_labels = class_labels.to(torch.device("cuda"))
            fake_samples = fake_samples.to(torch.device("cuda"))

        self.dropout = dropout
        self.train()
        self.optimizer.zero_grad()

        real_fake_output, class_output = self.forward(x)
        loss = self.compute_loss(real_fake_output, fake_samples, class_output, class_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters
    sequence_length = 10
    num_classes = 2
    vocab_size = 5000
    emb_dim = 128
    filter_sizes = [3, 4, 5]
    use_cuda = torch.cuda.is_available()  # Check if CUDA is available
    num_filters = [100, 100, 100]
    dropout = 0.5

    # Create discriminator model
    discriminator = Discriminator(sequence_length, num_classes, vocab_size, emb_dim, filter_sizes, use_cuda, num_filters, dropout)
    if use_cuda:
        discriminator.to(torch.device("cuda"))

    # Real data (batch_size, seq_len)
    real_data = torch.randint(0, vocab_size, (32, sequence_length))
    real_class_labels = torch.randint(0, num_classes, (32,))
    real_labels = torch.ones((32, 1), dtype=torch.float)  # Real data labels (all 1s)

    # Fake data (batch_size, seq_len)
    fake_data = torch.randint(0, vocab_size, (32, sequence_length))
    fake_class_labels = torch.randint(0, num_classes, (32,))
    fake_labels = torch.zeros((32, 1), dtype=torch.float)  # Fake data labels (all 0s)

    # Move data to GPU if available
    if use_cuda:
        real_data = real_data.to(torch.device("cuda"))
        real_labels = real_labels.to(torch.device("cuda"))
        real_class_labels = real_class_labels.to(torch.device("cuda"))
        fake_data = fake_data.to(torch.device("cuda"))
        fake_labels = fake_labels.to(torch.device("cuda"))
        fake_class_labels = fake_class_labels.to(torch.device("cuda"))

    # Forward pass for real data
    real_output, real_class_output = discriminator(real_data)
    real_loss = discriminator.compute_loss(real_output, fake_data, real_class_output, real_class_labels)

    # Forward pass for fake data
    fake_output, fake_class_output = discriminator(fake_data)
    fake_loss = discriminator.compute_loss(fake_output, real_data, fake_class_output, fake_class_labels)

    # Total loss
    total_loss = 0.5 * (real_loss + fake_loss)

    print(f"Real Loss: {real_loss.item():.4f}")
    print(f"Fake Loss: {fake_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")