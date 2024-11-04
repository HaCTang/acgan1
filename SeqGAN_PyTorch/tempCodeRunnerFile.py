    def train_step(self, x, class_label, rewards):
        """
        Performs a training step on the generator.
        Also known as unsupervised training.
        """
        self.train()
        hidden = self.init_hidden(x.size(0))
        x = x.to(self.emb.weight.device)
        class_label = class_label.to(self.class_emb.weight.device)
        rewards = rewards.to(self.emb.weight.device)
        hidden = tuple(h.to(self.emb.weight.device) for h in hidden)
        self.optimizer.zero_grad()

        # Forward pass
        logits, class_logits, _ = self.forward(x, class_label, hidden)
        logits = logits.view(-1, self.num_emb)  # [batch_size * seq_len, num_emb]
        target = x.view(-1)  # [batch_size * seq_len]

        # Calculate loss with rewards
        log_probs = F.log_softmax(logits, dim=-1)
        one_hot = F.one_hot(target, self.num_emb).float()
        token_loss = -torch.sum(rewards.view(-1, 1) * one_hot * log_probs) / self.batch_size

        # # Calculate class loss at the end of the sequence
        # _, class_logits, _ = self.forward(x, class_label, hidden)
        class_loss = self.auxiliary_loss(class_logits, class_label)

        loss = 0.5 * (token_loss + class_loss)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()