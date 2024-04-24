import torchvision

import matplotlib.pyplot as plt

def visualize_data(data, target):
    """
    Visualize data and target labels.

    Args:
        data (torch.Tensor): Tensor containing the image data to visualize.
        target (torch.Tensor): Tensor containing the corresponding labels for the images.

    Returns:
        None
    """
    # Create a 3x3 subplot with a figure size of 20x20
    fig, ax = plt.subplots(3, 3, figsize=(20, 20))

    # Iterate over the 9 subplots
    for i in range(9):
        # Display the i-th image in the current subplot
        ax[i // 3, i % 3].imshow(data[i].squeeze(0).numpy())

        # Set the title of the current subplot to the corresponding target label
        ax[i // 3, i % 3].set_title(target[i].item(), fontsize=50)

        # Turn off the axis for the current subplot
        ax[i // 3, i % 3].axis('off')

        # Turn off the grid for the current subplot
        ax[i // 3, i % 3].grid(False)

    # Save the visualization as an image file
    plt.savefig('figures/data_visualization.png')

    # Close the figure window
    plt.close()
    

# Visualize the reconstruction of a single image
def show_image(x, additional_info='') -> None:
    """Display a single image.

    Args:
        x (torch.Tensor): The image tensor to display.

    Returns:
        None
    """
    if x.dim() == 1:
        plt.imshow(x.view(28, 28).cpu().numpy(), cmap='gray')
    elif x.dim() == 3:
        plt.imshow(torchvision.utils.make_grid(x.cpu()).permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(f'figures/reconstruction_{additional_info}.png')
    
def visualize_loss(train_loss_list, test_loss_list):
    """
    Visualize training and test losses.

    Args:
        train_loss_list (list): List of training losses.
        test_loss_list (list): List of test losses.

    Returns:
        None
    """
    # Create a new figure
    plt.figure(figsize=(10, 5))

    # Plot the training loss curve
    plt.plot(train_loss_list, label='Training Loss', color='blue')

    # Plot the test loss curve
    plt.plot(test_loss_list, label='Test Loss', color='red')

    # Add a legend
    plt.legend()

    # Add x-axis label
    plt.xlabel('Epoch')

    # Add y-axis label
    plt.ylabel('Loss')

    # Add title
    plt.title('Training and Test Loss')

    # Save the visualization as an image file
    plt.savefig('figures/loss_visualization.png')

    # Close the figure window
    plt.close()
