"""
K-Means Segmentation Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to segment image using k-means clustering.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are allowed to add your own functions if needed.
You should design you algorithm as fast as possible. To avoid repetitve calculation, you are suggested to depict clustering based on statistic histogram [0,255]. 
You will be graded based on the total distortion, e.g., sum of distances, the less the better your clustering is.
"""


import utils
import numpy as np
import json
import time


def kmeans(img,k):
    """
    Implement kmeans clustering on the given image.
    Steps:
    (1) Random initialize the centers.
    (2) Calculate distances and update centers, stop when centers do not change.
    (3) Iterate all initializations and return the best result.
    Arg: Input image;
         Number of K. 
    Return: Clustering center values;
            Clustering labels of all pixels;
            Minimum summation of distance between each pixel and its center.  
    """
    def forward(img, centers):
        new_c1 = np.where(abs(img - centers[0]) <= abs(img - centers[1]), img, 0)
        count = len(new_c1[new_c1 != 0])
        if count == 0:
            count = 1
        new_c1 = np.sum(new_c1) / count

        new_c2 = np.where(abs(img - centers[0]) > abs(img - centers[1]), img, 0)
        count = len(new_c2[new_c2 != 0])
        if count == 0:
            count = 1
        new_c2 = np.sum(new_c2) / count

        return [new_c1, new_c2]

    def flatten(a):
        return [i for nested in a for i in nested]

    def get_combinations(img):
        return [(img[i], img[j]) for i in range(len(img)) for j in range(i + 1, len(img))]

    flattened_img = flatten(img)
    all_combinations = get_combinations(np.unique(flattened_img))
    already_seen = []
    img = img.astype(int)
    saved_models = []
    for epoch, old_centers in enumerate(all_combinations):
        if epoch % 500 == 0:
            print('Epoch: ', epoch)

        flag = False
        k_means_iteration = 10
        for j in range(k_means_iteration):
            new_centers = forward(img, old_centers)
            if new_centers[0] == old_centers[0] and new_centers[1] == old_centers[1]:
                break
            else:
                old_centers = new_centers
            if new_centers in already_seen:
                flag = True
                break
            else:
                already_seen.append(new_centers)

        if not flag:
            error = np.sum(np.where(abs(img - new_centers[0]) <= abs(img - new_centers[1]), abs(img - new_centers[0]), abs(img - new_centers[1])))
            saved_models.append([new_centers, error])

    new_centers, error = min(saved_models, key=lambda x: x[1])

    cluster_labels = np.where(abs(img - new_centers[0]) <= abs(img - new_centers[1]), 0, 1)
    cluster_labels = cluster_labels.astype(np.uint8)

    return new_centers, cluster_labels, error
    # TODO: implement this function.


def visualize(centers,labels):
    """
    Convert the image to segmentation map replacing each pixel value with its center.
    Arg: Clustering center values;
         Clustering labels of all pixels. 
    Return: Segmentation map.
    """
    return np.where(labels == 0, centers[0], centers[1]).astype(np.uint8)
    # TODO: implement this function.

     
if __name__ == "__main__":
    img = utils.read_image('lenna.png')
    k = 2

    start_time = time.time()
    centers, labels, sumdistance = kmeans(img,k)
    result = visualize(centers, labels)
    end_time = time.time()

    running_time = end_time - start_time
    print(running_time)

    centers = list(centers)
    with open('results/task1.json', "w") as jsonFile:
        jsonFile.write(json.dumps({"centers":centers, "distance":sumdistance, "time":running_time}))
    utils.write_image(result, 'results/task1_result.jpg')
