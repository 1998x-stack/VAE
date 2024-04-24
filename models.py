import torch
from torch import nn


"""
    一个简单的高斯MLP编码器和解码器实现
"""

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        初始化编码器

        参数:
            input_dim (int): 输入维度
            hidden_dim (int): 隐藏层维度
            latent_dim (int): 潜在空间维度
        """
        super(Encoder, self).__init__()
        self.linear_layer1 = nn.Linear(input_dim, hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)

        self.activation = nn.LeakyReLU(0.1)
        self.training =  True

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入张量

        返回:
            mu (torch.Tensor): 均值张量
            log_var (torch.Tensor): 对数方差张量
        """
        h1 = self.activation(self.linear_layer1(x))
        h2 = self.activation(self.linear_layer2(h1))
        mu = self.mu_layer(h2)
        log_var = self.log_var_layer(h2) # log_var \in (-inf, inf) while var \in [0, inf)
        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim):
        """
        初始化解码器

        参数:
            latent_dim (int): 潜在空间维度
            hidden_dim (int): 隐藏层维度
            input_dim (int): 输入维度
        """
        super(Decoder, self).__init__()
        self.linear_layer1 = nn.Linear(latent_dim, hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, input_dim)

        self.activation = nn.LeakyReLU(0.1)
        self.training =  True

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入张量

        返回:
            out (torch.Tensor): 输出张量
        """
        h1 = self.activation(self.linear_layer1(x))
        h2 = self.activation(self.linear_layer2(h1))
        out = nn.Sigmoid()(self.out_layer(h2))
        return out
    

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        """
        初始化VAE模型

        参数:
            encoder (Encoder): 编码器实例
            decoder (Decoder): 解码器实例
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reparameter(self, mu, log_var):
        """
        重参数化技巧

        参数:
            mu (torch.Tensor): 均值张量
            log_var (torch.Tensor): 对数方差张量

        返回:
            z (torch.Tensor): 重参数化后的潜在向量
        """
        var = torch.exp(0.5 * log_var)  # sigma = exp(0.5 * log(sigma^2))
        epsilon = torch.randn_like(var).to(self.DEVICE)
        z = mu + var * epsilon  # mu + std * N(0,1) ~ N(mu, std)
        return z

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入张量

        返回:
            x_hat (torch.Tensor): 重构输出张量
            mu (torch.Tensor): 均值张量
            log_var (torch.Tensor): 对数方差张量
        """
        mu, log_var = self.encoder(x)
        z = self.reparameter(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var