"""
Template Matching
(Due date: Sep. 25, 3 P.M., 2019)

The goal of this task is to experiment with template matching techniques, i.e., normalized cross correlation (NCC).

Please complete all the functions that are labelled with '# TODO'. When implementing those functions, comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in 'utils.py'
and the functions you implement in 'task1.py' are of great help.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
"""


import argparse
import json
import os

import utils
from task1 import *


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task2.jpg",
        help="path to the image")
    parser.add_argument(
        "--template-path",
        type=str,
        default="./data/proj1-task2-template.jpg",
        help="path to the template"
    )
    parser.add_argument(
        "--result-saving-path",
        dest="rs_path",
        type=str,
        default="./results/task2.json",
        help="path to file which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args

def norm_xcorr2d(patch, template):
    """Computes the NCC value between a image patch and a template.

    The image patch and the template are of the same size. The formula used to compute the NCC value is:
    sum_{i,j}(x_{i,j} - x^{m}_{i,j})(y_{i,j} - y^{m}_{i,j}) / (sum_{i,j}(x_{i,j} - x^{m}_{i,j}) ** 2 * sum_{i,j}(y_{i,j} - y^{m}_{i,j})) ** 0.5
    This equation is the one shown in Prof. Yuan's ppt.

    Args:
        patch: nested list (int), image patch.
        template: nested list (int), template.

    Returns:
        value (float): the NCC value between a image patch and a template.
    """
    def sum_matrix(matrix):
        return sum([e for i in matrix for e in i])
    def element_sub(matrix, value):
        return [i-value for row in matrix for i in row]
    def element_mul(a, b):
        return [a[i]*b[i] for i, row in enumerate(a)]
    def mean(matrix):
        num_elements = len(matrix) * len(matrix[0])
        return sum_matrix(matrix) / num_elements

    diff_patch = element_sub(patch, mean(patch))
    diff_template = element_sub(template, mean(template))
    numerator = sum(element_mul(diff_patch, diff_template))

    sum_sqr_diff_patch = sum([i**2 for i in diff_patch])
    sum_sqr_diff_template = sum([i**2 for i in diff_template])
    denominator = np.sqrt(sum_sqr_diff_patch * sum_sqr_diff_template)
    return numerator / denominator
    # raise NotImplementedError


def match(img, template):
    """Locates the template, i.e., a image patch, in a large image using template matching techniques, i.e., NCC.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        x (int): row that the character appears (starts from 0).
        y (int): column that the character appears (starts from 0).
        max_value (float): maximum NCC value.
    """
    # TODO: implement this function.
    template_dims = (len(template), len(template[0]))
    img_dims = (len(img), len(img[0]))
    loop_dims = (img_dims[0] - template_dims[0], img_dims[1] - template_dims[1])
    output_img = [[0.0 for i in range(loop_dims[1])] for j in range(loop_dims[0])]
    for i in range(loop_dims[0]):
        for j in range(loop_dims[1]):
            cropped_img = utils.crop(img, i, i+template_dims[0], j, j+template_dims[1])
            output_img[i][j] = norm_xcorr2d(cropped_img, template)
    print (output_img)
    max_element = max([(i, j, val) for i, inner_loop in enumerate(output_img) for j, val in enumerate(inner_loop)], key = lambda x: x[2])
    print (max_element)
    return max_element
    # raise NotImplementedError

def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    # template = utils.crop(img, xmin=10, xmax=30, ymin=10, ymax=30)
    # template = np.asarray(template, dtype=np.uint8)
    # cv2.imwrite("./data/proj1-task2-template.jpg", template)
    template = read_image(args.template_path)
    x, y, max_value = match(img, template)
    # The correct results are: x: 17, y: 129, max_value: 0.994
    with open(args.rs_path, "w") as file:
        json.dump({"x": x, "y": y, "value": max_value}, file)


if __name__ == "__main__":
    main()
