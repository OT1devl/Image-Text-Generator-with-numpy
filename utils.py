import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import glob
import imageio

def one_hot(data, n_classes):
    new_data = np.zeros((data.shape[0], n_classes))
    new_data[np.arange(data.shape[0]), data] = 1
    return new_data

def generate_and_save_images(generator, seed, gen_loss, disc_loss, epoch, imgs_path):
    clear_output(wait=True)
    generated_imgs = generator.forward(*seed)
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
    rows = cols = int(seed[0].shape[0]**0.5)
    if rows * cols < seed[0].shape[0]:
        cols += 1
    fig, axes = plt.subplots(rows, cols, figsize=(6, 6))
    fig.suptitle(f'Epoch: {epoch}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}', fontsize=12)
    for i, ax in enumerate(axes.flat):
        if i < seed[0].shape[0]:
            ax.imshow(generated_imgs[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.savefig(os.path.join(imgs_path, f'generated_images_epoch_{epoch}.png'))
    plt.show()

def make_gif(path):
    pattern = os.path.join(path, 'generated_images_epoch_*.png')
    img_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    images = [imageio.imread(img) for img in img_files]
    gif_filename = os.path.join(path, 'training.gif')
    imageio.mimsave(gif_filename, images, duration=0.5)
    print("GIF saved as:", gif_filename)