"""
RANSAC Algorithm Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to fit a line to the given points using RANSAC algorithm, and output
the names of inlier points and outlier points for the line.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
You can use the library random
Hint: It is recommended to record the two initial points each time, such that you will Not 
start from this two points in next iteration.
"""
import random


def solution(input_points, t, d, k):
    """
    :param input_points:
           t: t is the perpendicular distance threshold from a point to a line
           d: d is the number of nearby points required to assert a model fits well, you may not need this parameter
           k: k is the number of iteration times
           Note that, n for line should be 2
           (more information can be found on the page 90 of slides "Image Features and Matching")
    :return: inlier_points_name, outlier_points_name
    inlier_points_name and outlier_points_name is two list, each element of them is str type.
    For example: If 'a','b' is inlier_points and 'c' is outlier_point.
    the output should be two lists of ['a', 'b'], ['c'].
    Note that, these two lists should be non-empty.
    """
    # TODO: implement this function.
    def rnd():
        exp = random.randint(-19, -5)
        base = 0.9 * random.random() + 0.1
        return base * 10**exp

    def create_line(points):

        p1 = points[0]['value']
        p2 = points[1]['value']

        slope = (p2[1] - p1[1]) / (p2[0] - p1[0] + rnd())
        intercept = -slope*p2[0] + p2[1]

        return {'slope': slope, 'intercept': intercept}

    def dist_from_line(point, line):
        num = (line['slope']*point[0] - point[1] + line['intercept'])
        denum = 1 + line['slope'] ** 2
        return abs(num / denum ** 0.5)

    checked = []
    min_error = 10**20
    num_combinations = len(input_points) * (len(input_points) - 1) / 2
    count = 0

    while len(checked) < num_combinations * 2 and count < k:
        inliers = []
        outliers = []
        error = 0
        random.shuffle(input_points)
        selected_points = input_points[:2]
        unselected_points = input_points[2:]

        name_of_points = [selected_points[0]['name'], selected_points[1]['name']]

        if name_of_points not in checked:
            checked.append(name_of_points)
            checked.append(name_of_points[::-1])
            count = count + 1
            # print ('Count', count)
        else:
            continue

        line = create_line(selected_points)

        inliers.append(selected_points[0]['name'])
        inliers.append(selected_points[1]['name'])

        for point in unselected_points:
            dist = dist_from_line(point['value'], line)
            if dist <= t:
                error += dist
                inliers.append(point['name'])
            else:
                outliers.append(point['name'])

        if len(inliers) - 2 >= d:
            error = error / (len(inliers) - 2)
            if error < min_error:
                min_error = error
                output = (inliers, outliers)

    return output


if __name__ == "__main__":
    input_points = [{'name': 'a', 'value': (0.0, 1.0)}, {'name': 'b', 'value': (2.0, 1.0)},
                    {'name': 'c', 'value': (3.0, 1.0)}, {'name': 'd', 'value': (0.0, 3.0)},
                    {'name': 'e', 'value': (1.0, 2.0)}, {'name': 'f', 'value': (1.5, 1.5)},
                    {'name': 'g', 'value': (1.0, 1.0)}, {'name': 'h', 'value': (1.5, 2.0)}]
    t = 0.5
    d = 3
    k = 100
    inlier_points_name, outlier_points_name = solution(input_points, t, d, k)  # TODO
    assert len(inlier_points_name) + len(outlier_points_name) == 8  
    f = open('./results/task1_result.txt', 'w')
    f.write('inlier points: ')
    for inliers in inlier_points_name:
        f.write(inliers + ',')
    f.write('\n')
    f.write('outlier points: ')
    for outliers in outlier_points_name:
        f.write(outliers + ',')
    f.close()


