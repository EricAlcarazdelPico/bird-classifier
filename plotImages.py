import matplotlib.pyplot as plt


def plotImages(images_arr):
    fig_size = 4 * len(images_arr)
    fig, axes = plt.subplots(1, len(images_arr), figsize=(fig_size, fig_size))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
