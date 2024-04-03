import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        r_out, _ = self.rnn(x)
        output = torch.sigmoid(self.fc(r_out))
        
        return output


class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        r_out, _ = self.rnn(h)
        output = torch.sigmoid(self.fc(r_out))
        return output


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z):
        r_out, _ = self.rnn(z)
        output = torch.sigmoid(self.fc(r_out))
        return output


class Supervisor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers-1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        r_out, _ = self.rnn(h)
        output = torch.sigmoid(self.fc(r_out))
        return output

    
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, h):
        r_out, _ = self.rnn(h)
        output = torch.sigmoid(self.fc(r_out))
        return output


class TimeGAN(nn.Module):
    def __init__(self, 
                 dataloader, 
                 input_dim = 1, 
                 output_dim = 1, 
                 hidden_dim = 24, 
                 num_layers = 3) -> None:
        super().__init__()
        
        self.dl = dataloader
        self.hidden_dim = hidden_dim
        self.embedder = Embedder(input_dim, hidden_dim, num_layers)
        self.recovery = Recovery(hidden_dim, output_dim, num_layers)
        self.generator = Generator(hidden_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, hidden_dim, num_layers-1)
        self.discriminator = Discriminator(hidden_dim, hidden_dim, num_layers)
        
        # lossfunction
        self.E_mseloss = nn.MSELoss()
        self.S_mseloss = nn.MSELoss()

        # 옵티마이저
        self.E_optimizer = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=0.001)
        self.G_optimizer = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=0.001)
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)

    def train(self, 
              num_epochs_embedding = 10,
              num_epochs_adversarial = 10,
              num_epochs_supervised = 10,
              ):
        
        # Embedding Phase: Training Embedder and Recovery
        for epoch in range(num_epochs_embedding):
            with tqdm(self.dl) as pbar:
                for X in pbar:
                    H = self.embedder(X)
                    X_tilde = self.recovery(H)
                    E_loss = self.E_mseloss(X, X_tilde)
                    
                    self.E_optimizer.zero_grad()
                    E_loss.backward()
                    self.E_optimizer.step()
                    
                    # visualize
                    pbar.set_description(desc=f"Epoch {epoch+1}/{num_epochs_embedding} E_loss : {E_loss:0.6f}")


        # Supervised Training Phase
        # Here, the intention was to train the generator with supervised loss,
        # but it mistakenly updates using the embedder optimizer.
        # Adjusting to reflect the correct training intention.
        for epoch in range(num_epochs_supervised):
            with tqdm(self.dl) as pbar:
                for X in pbar:
                    Z = torch.randn(X.shape[:-1] + (self.hidden_dim,))
                    H = self.embedder(X)
                    E_hat = self.generator(Z)
                    H_hat = self.supervisor(E_hat)  # Generate supervised embeddings
                    S_loss = self.S_mseloss(H, H_hat)

                    self.G_optimizer.zero_grad()
                    S_loss.backward()
                    self.G_optimizer.step()
                    pbar.set_description(desc=f"Epoch {epoch+1}/{num_epochs_supervised} S_loss : {S_loss:0.5f}")

        # Adversarial Training Phase
        for epoch in range(num_epochs_adversarial):
            with tqdm(self.dl) as pbar:
                for X in pbar:
                    # Discriminator Training
                    for _ in range(2):  # Typically, discriminator is trained more frequently.
                        Z = torch.randn(X.shape[:-1] + (self.hidden_dim,))
                        H = self.embedder(X)
                        H_fake = self.generator(Z).detach()  # Detach to avoid training generator here.

                        D_real = self.discriminator(H)
                        D_fake = self.discriminator(H_fake)

                        D_loss = -torch.mean(torch.log(D_real + 1e-8) + torch.log(1 - D_fake + 1e-8))

                        self.D_optimizer.zero_grad()
                        D_loss.backward()
                        self.D_optimizer.step()
                            
                    # Generator Training
                    Z = torch.randn(X.shape[:-1] + (self.hidden_dim,))
                    H_fake = self.generator(Z)
                    D_fake = self.discriminator(H_fake)
                    G_loss = -torch.mean(torch.log(D_fake + 1e-8))

                    self.G_optimizer.zero_grad()
                    G_loss.backward()
                    self.G_optimizer.step()
                    
                    pbar.set_description(f"Epochs {epoch+1} / {num_epochs_adversarial} G_loss : {G_loss:0.5f} D_loss : {D_loss:0.5f}")
                        
    def forward(self, Z):
        H_fake = self.generator(Z)
        X_fake = self.recovery(H_fake).detach()

        return X_fake

    
        
    