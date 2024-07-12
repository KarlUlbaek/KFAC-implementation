import numpy as np
import matplotlib.pyplot as plt
import torch



def display(before, after=None, num_examples=4, image_dim=28, close = True, vmax=1):
    if not isinstance(before, np.ndarray):
        before = before.cpu().detach().numpy()

    if not isinstance(after, np.ndarray) and after is not None:
        after = after.cpu().detach().numpy()

    fig, ax = plt.subplots(2 if after is not None else 1, num_examples)

    if after is not None:
        for i in range(num_examples):
            example = before[i,:].reshape(image_dim, image_dim)
            ax[0, i].imshow(example, cmap='gray', vmin=0, vmax=1)

            example = after[i, :].reshape(image_dim, image_dim)
            ax[1, i].imshow(example, cmap='gray', vmin=0, vmax=1)

        fig.suptitle("Before and after reconstruction")

    else:
        for i in range(num_examples):
            example = before[i,:].reshape(image_dim, image_dim)
            ax[i].imshow(example, cmap='gray', vmin=0, vmax=vmax)
        fig.suptitle("Data")

    #plt.tight_layout()

    if close:
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    x = np.load("data/digits_train.npy")
    x = x / 255

    display(x, x)







