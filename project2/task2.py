"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    def find_features(image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        descriptor = cv2.xfeatures2d.SIFT_create()
        kps, features = descriptor.detectAndCompute(image, None)

        kps = np.float32([kp.pt for kp in kps])
        return kps, features

    def matchKeypoints(kps_left, kps_right, featuresA, featuresB, ratio, threshold):

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = [(m[0].trainIdx, m[0].queryIdx) for m in matches if len(m) == 2 and m[0].distance < m[1].distance*0.8]

        points_left = np.float32([kps_left[i] for (_, i) in matches])
        points_right = np.float32([kps_right[i] for (i, _) in matches])

        (H, flag) = cv2.findHomography(points_left, points_right, cv2.RANSAC, threshold)

        return matches, H, flag

    kps_left, features_left = find_features(left_img)
    kps_right, features_right = find_features(right_img)

    (matches, H, flag) = matchKeypoints(kps_right, kps_left, features_right, features_left, 1, 4.0)

    result = cv2.warpPerspective(right_img, H, (right_img.shape[1] + left_img.shape[1], right_img.shape[0]))
    result[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    return result

    # raise NotImplementedError


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg',result_image)


