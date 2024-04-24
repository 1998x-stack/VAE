import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from data_visualizer import visualize_data


def get_mnist_loader(batch_size, dataset_path = '~/datasets'):
    """
    获取MNIST数据集的数据加载器

    参数:
        batch_size (int): 每个批次的样本数量
        dataset_path (str): 数据集存储路径，默认为'~/datasets'

    返回值:
        train_loader (torch.utils.data.DataLoader): 训练集的数据加载器
        test_loader (torch.utils.data.DataLoader): 测试集的数据加载器
    """

    # 定义MNIST数据集的转换操作
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(),]
    )

    # 定义数据加载器的参数
    kwargs = {'num_workers': 1, 'pin_memory': True} 

    # 创建训练集和测试集的数据集对象
    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    # 创建训练集和测试集的数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

if __name__ == '__main__':
    # 获取训练集和测试集的数据加载器
    train_loader, test_loader = get_mnist_loader(32)
    print(len(train_loader))
    print(len(test_loader))
    for i, (data, target) in enumerate(train_loader):
        print(data.shape, target.shape)
        break
    visualize_data(data, target)
    
    for i, (data, target) in enumerate(test_loader):
        print(data.shape, target.shape)
        break