import torch
from torch import nn, optim
from models import Encoder, Decoder, VAE
from config import CONFIG
from dataset import get_mnist_loader
from data_visualizer import show_image, visualize_loss
    
# 损失函数和优化器
def loss_function(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """计算VAE损失，包括重建误差和KL散度。

    Args:
        x (torch.Tensor): 原始输入数据。
        x_hat (torch.Tensor): 重建的输入数据。
        mu (torch.Tensor): 编码器的均值。
        log_var (torch.Tensor): 编码器的对数方差。

    Returns:
        torch.Tensor: 总损失。
    """
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KL_loss = 0.5 * torch.sum(mu.pow(2) + torch.exp(log_var) - 1 - log_var)
    return reproduction_loss + KL_loss

# 训练过程
def train(model: nn.Module, data_loader, optimizer: optim.Optimizer, config) -> None:
    """训练VAE模型一个epoch。

    Args:
        model (nn.Module): VAE模型。
        data_loader (DataLoader): 数据集的DataLoader。
        optimizer (optim.Optimizer): 模型的优化器。
        config (Config): 配置对象。

    Returns:
        None
    """
    model.train()
    total_loss = 0
    for batch_idx, (x, _) in enumerate(data_loader):
        x = x.view(-1, config.x_dim)  # 将图像展平
        x = x.to(config.DEVICE)
        optimizer.zero_grad()
        x_hat, mu, log_var = model(x)
        loss = loss_function(x, x_hat, mu, log_var)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    average_loss = total_loss / len(data_loader.dataset)
    print(f"平均损失: {average_loss:.4f}")
    return average_loss

# 测试过程
def test(model: nn.Module, data_loader, config) -> None:
    """评估VAE在测试数据集上的性能。

    Args:
        model (nn.Module): VAE模型。
        data_loader (DataLoader): 数据集的DataLoader。
        config (Config): 配置对象。

    Returns:
        None
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.view(-1, config.x_dim)  # 将图像展平
            x = x.to(config.DEVICE)
            x_hat, mu, log_var = model(x)
            loss = loss_function(x, x_hat, mu, log_var)
            total_loss += loss.item()
    average_loss = total_loss / len(data_loader.dataset)
    print(f"测试集平均损失: {average_loss:.4f}")
    return average_loss

if __name__ == "__main__":
    config = CONFIG()
    encoder = Encoder(input_dim=config.x_dim, hidden_dim=config.hidden_dim, latent_dim=config.latent_dim)
    decoder = Decoder(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, input_dim=config.x_dim)
    model = VAE(encoder=encoder, decoder=decoder).to(config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    train_loader, test_loader = get_mnist_loader(batch_size=config.batch_size)

    train_loss_list, test_loss_list = [], []
    # 训练和测试循环
    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}")
        train_loss = train(model, train_loader, optimizer, config)
        test_loss = test(model, test_loader, config)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    print("训练完成！")
    visualize_loss(train_loss_list, test_loss_list)
    # 显示一些重建的图片
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(config.DEVICE)
    with torch.no_grad():
        reconstructed, _, _ = model(test_images.view(-1, config.x_dim))
    show_image(reconstructed[0], 'reconstruct')
    show_image(test_images[0], 'original')