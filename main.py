import wandb  # Ensure you've initialized wandb and logged in appropriately.

def train_GAN(num_epochs_embedding, 
              num_epochs_supervised, 
              num_epochs_adversarial, 
              data_loader, 
              batch_size, 
              seq_len, 
              hidden_dim,
              embedder, 
              recovery, 
              generator, 
              supervisor, 
              discriminator, 
              E_optimizer, 
              G_optimizer,
              D_optimizer):

    # Embedding Phase: Training Embedder and Recovery
    for epoch in range(num_epochs_embedding):
        for X_mb, T_mb in data_loader:  # Iterate through mini-batches from the data loader.
            H_mb = embedder(X_mb)
            X_tilde = recovery(H_mb)
            E_loss = torch.mean((X_mb - X_tilde)**2)  # MSE loss

            E_optimizer.zero_grad()
            E_loss.backward()
            E_optimizer.step()
            wandb.log({"Embedding Loss": E_loss})
            
    # Supervised Training Phase
    # Here, the intention was to train the generator with supervised loss,
    # but it mistakenly updates using the embedder optimizer.
    # Adjusting to reflect the correct training intention.
    for epoch in range(num_epochs_supervised):
        for X_mb, T_mb in data_loader:
            Z_mb = torch.randn([batch_size, seq_len, hidden_dim])
            H_mb = embedder(X_mb)
            E_hat = generator(Z_mb)
            H_hat_supervised = supervisor(E_hat)  # Generate supervised embeddings
            S_loss = torch.mean((H_mb - H_hat_supervised)**2)  # Supervised loss for generator

            G_optimizer.zero_grad()
            S_loss.backward()
            G_optimizer.step()
            wandb.log({"Supervised Loss": S_loss})
            
    # Adversarial Training Phase
    for epoch in range(num_epochs_adversarial):
        for X_mb, T_mb in data_loader:
            # Discriminator Training
            for _ in range(2):  # Typically, discriminator is trained more frequently.
                Z_mb = torch.randn([batch_size, seq_len, hidden_dim])
                H_mb = embedder(X_mb)
                H_fake = generator(Z_mb).detach()  # Detach to avoid training generator here.

                D_real = discriminator(H_mb)
                D_fake = discriminator(H_fake)

                D_loss = -torch.mean(torch.log(D_real + 1e-8) + torch.log(1 - D_fake + 1e-8))

                D_optimizer.zero_grad()
                D_loss.backward()
                D_optimizer.step()
                wandb.log({"Discriminator Loss": D_loss})
                
            # Generator Training
            Z_mb = torch.randn([batch_size, seq_len, hidden_dim])
            H_fake = generator(Z_mb)

            D_fake = discriminator(H_fake)
            G_loss = -torch.mean(torch.log(D_fake + 1e-8))

            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            wandb.log({"Generator Loss": G_loss})
            
    return embedder, recovery, generator, supervisor, discriminator

