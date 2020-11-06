"""
Denoise Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to denoise image using median filter.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are suggested to use utils.zero_pad.
"""


import utils
import numpy as np
import json

def median_filter(img):
    """
    Implement median filter on the given image.
    Steps:
    (1) Pad the image with zero to ensure that the output is of the same size as the input image.
    (2) Calculate the filtered image.
    Arg: Input image. 
    Return: Filtered image.
    """
    # Referenced from project1
    def crop(img, xmin, xmax, ymin, ymax):
        """Crops a given image."""
        if len(img) < xmax:
            print('WARNING')
        patch = img[xmin: xmax]
        patch = [row[ymin: ymax] for row in patch]
        return patch
    kernel_size = 3
    pad_size = kernel_size // 2
    img_dims = [len(img), len(img[0])]
    padded_img = utils.zero_pad(img, pad_size, pad_size)
    output_img = np.copy(img)
    for i in range(img_dims[0]):
        for j in range(img_dims[1]):
            cropped_img = crop(padded_img, i, i+kernel_size, j, j+kernel_size)
            output_img[i][j] = int(np.median(cropped_img))
    return output_img
    # TODO: implement this function.


def mse(img1, img2):
    """
    Calculate mean square error of two images.
    Arg: Two images to be compared.
    Return: Mean square error.
    """
    return float(np.sum((img1 - img2)**2)) / (img1.shape[0] * img1.shape[1])
    # TODO: implement this function.
    

if __name__ == "__main__":
    img = utils.read_image('lenna-noise.png')
    gt = utils.read_image('lenna-denoise.png')

    result = median_filter(img)
    error = mse(gt, result)

    with open('results/task2.json', "w") as file:
        json.dump(error, file)
    utils.write_image(result,'results/task2_result.jpg')


