import torch
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    """Simple MLP autoencoder baseline. Flattens the sequence window into a single vector."""
    def __init__(self, seq_len=20, input_dim=9, hidden_dim=128, latent_dim=32):
        super(DenseAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        flat_dim = seq_len * input_dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)       # (batch, seq_len * input_dim)
        latent = self.encoder(x_flat)
        recon_flat = self.decoder(latent)
        recon = recon_flat.view(batch_size, self.seq_len, self.input_dim)
        return recon


class GRUAutoencoder(nn.Module):
    def __init__(self, seq_len=20, input_dim=9, hidden_dim=64, latent_dim=16, num_layers=2):
        super(GRUAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder GRU
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        out, _ = self.encoder_gru(x, h0)
        last_out = out[:, -1, :]  # Shape: (batch, hidden_dim)
        latent = self.fc_latent(last_out)  # Shape: (batch, latent_dim)
        
        # Decode
        h_dec = self.decoder_fc(latent)
        h_dec = h_dec.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        
        latent_seq = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, latent_dim)
        dec_input = self.decoder_fc(latent_seq)
        
        out_dec, _ = self.decoder_gru(dec_input, h_dec)
        recon_x = self.decoder_out(out_dec)
        return recon_x


class TCNAutoencoder(nn.Module):
    def __init__(self, seq_len=20, input_dim=9, hidden_dim=64, latent_dim=16, kernel_size=3):
        super(TCNAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Encoder (using Conv1d)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=latent_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        
        # Decoder (using Conv1d)
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=latent_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=kernel_size, padding=kernel_size//2)
        )
        
    def forward(self, x):
        # Input shape: (batch, seq_len, input_dim)
        x_trans = x.transpose(1, 2)
        latent = self.encoder(x_trans)
        recon_trans = self.decoder(latent)
        recon_x = recon_trans.transpose(1, 2)
        return recon_x


class CNNGRUAutoencoder(nn.Module):
    """CNN extracts local features, GRU handles temporal dependencies."""
    def __init__(self, seq_len=20, input_dim=9, hidden_dim=64, latent_dim=16, num_layers=2):
        super(CNNGRUAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder: CNN then GRU
        self.enc_conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.enc_relu = nn.ReLU()
        self.enc_gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: GRU then CNN
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.dec_gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.dec_conv = nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode CNN
        x_trans = x.transpose(1, 2)
        conv_out = self.enc_conv(x_trans)
        conv_out = self.enc_relu(conv_out)
        conv_out = conv_out.transpose(1, 2)
        
        # Encode GRU
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        gru_out, _ = self.enc_gru(conv_out, h0)
        last_out = gru_out[:, -1, :]
        latent = self.fc_latent(last_out)
        
        # Decode GRU
        h_dec = self.dec_fc(latent)
        h_dec_state = h_dec.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        latent_seq = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_input = self.dec_fc(latent_seq)
        
        dec_gru_out, _ = self.dec_gru(dec_input, h_dec_state)
        
        # Decode CNN
        dec_gru_out = dec_gru_out.transpose(1, 2)
        recon_trans = self.dec_conv(dec_gru_out)
        recon_x = recon_trans.transpose(1, 2)
        return recon_x
