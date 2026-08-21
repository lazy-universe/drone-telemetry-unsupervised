from implement.utils.deep_learning.autoencoders import (
    DenseAutoencoder,
    GRUAutoencoder,
    TCNAutoencoder,
    CNNGRUAutoencoder
)

def get_unsupervised_models(input_dim=9, seq_len=20):
    """
    Returns a dictionary of deep learning autoencoders for unsupervised sequence anomaly detection.
    """
    return {
        'Dense Autoencoder':   DenseAutoencoder(seq_len=seq_len, input_dim=input_dim),
        'GRU Autoencoder':     GRUAutoencoder(seq_len=seq_len, input_dim=input_dim),
        'TCN Autoencoder':     TCNAutoencoder(seq_len=seq_len, input_dim=input_dim),
        'CNN-GRU Autoencoder': CNNGRUAutoencoder(seq_len=seq_len, input_dim=input_dim),
    }
