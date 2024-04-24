import torch
class CONFIG:
    def __init__(self) -> None:
        self.batch_size = 128
        self.epochs = 20
        self.lr = 0.001
        self.hidden_dim = 400
        self.latent_dim = 20
        self.x_dim = 784
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'