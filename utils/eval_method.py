import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist


def gaussian_eval(gt_points, dt_points, scalar=32, value_only=False):
    """
    gt_points: ground true car pos
    dt_points: predicted true car pos
    scalar: (scalar=𝝈^𝟐) covariance matrix of 2D Gaussian
    """

    # get ground true car pos
    valid_gt_points = [point for point in gt_points if point[2] != -1]
    car_coordinates = [point[:2] for point in valid_gt_points]
    gt_num_car = len(car_coordinates)

    # get predict car pos: center of two light, if only one light, assume light pos = car pos
    dt_car_pos = []
    for points in dt_points:
        if min(points[0]) == -1:
            midpoint = [points[1][0], points[1][1]]
        elif min(points[1]) == -1:
            midpoint = [points[0][0], points[0][1]]
        else:
            midpoint = [(points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2]
        dt_car_pos.append(midpoint)

    dt_num_car = len(dt_car_pos)
    """
    no car in img but model predict "car":
    when true=0 
    predict=8->eval_value=0
    predict=0->eval_value=1
    so eval_value = -1/8 * len(dt_car_pos) + 1
    
    has car in img but model predict "no car":
    when predict=0
    true=0 -> eval_value=1
    true=8 -> eval_value=0
    """
    nearest_positions = []
    gaussian_function_list = []
    vi = []
    dt_car_pos_copy = dt_car_pos
    if gt_num_car == 0:
        eval_result = -1 / 8 * dt_num_car + 1
    elif dt_num_car == 0:
        eval_result = -1 / 8 * gt_num_car + 1
    else:
        # find nearest_positions between ground true and predict car
        while len(dt_car_pos) < gt_num_car:
            dt_car_pos.append([0, 0])
        for car_coordinate in car_coordinates:
            # compute distance
            distances = cdist([car_coordinate], dt_car_pos)
            nearest_index = np.argmin(distances)
            nearest_pos = dt_car_pos[nearest_index]
            dt_car_pos = dt_car_pos[:nearest_index] + dt_car_pos[nearest_index + 1:]
            nearest_positions.append(nearest_pos)

            # for each ground true car, create 2D gaussian function
            mean = car_coordinate
            cov1 = np.eye(2)*scalar  # the Gaussian distribution becomes flatter when scalar higher
            mvn = multivariate_normal(mean, cov1)
            max_value = mvn.pdf(mean)
            dt_value = mvn.pdf(nearest_pos)

            normalized_value = dt_value / max_value  # normalized -> (0,1)
            """
            print('car_coordinate', car_coordinate)
            print('nearest_pos', nearest_pos)
            print('max', max_value)
            print('dt_value', dt_value)
            print('normalized_value', normalized_value)
            """
            gaussian_function_list.append(mvn)
            vi.append(normalized_value)
        eval_result = sum(vi) / max(dt_num_car, gt_num_car)  # % by number of dt car
    if not value_only:
        return eval_result, dt_car_pos_copy, nearest_positions, car_coordinates, vi
    else:
        return eval_result


def gaussian_method(batch_tags_results, keypoints, batch_size, scalar=128*128):
    keypoints = keypoints.tolist()
    # for each img
    eval_results = 0

    for i in range(batch_size):
        gt_points = keypoints[i][0]
        dt_points = batch_tags_results[i]
        eval_value = gaussian_eval(gt_points, dt_points, scalar=scalar, value_only=True)
        eval_results += eval_value

    mean_eval_score_batch = eval_results/batch_size
    return mean_eval_score_batch


if __name__ == '__main__':
    # example/test for gaussian_method
    # gt_points = [(left light1), (right light1) , (left light2), (right light2)]
    gt_points = [[64, 64, 1], [60, 60, 1], [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]]
    dt_points = [[(64, 64), (64, 64)], [(0, 0), (0, 0)], [(40, 40), (-1, -1)]]  # single light (-1,-1)
    eval_value = gaussian_eval(gt_points, dt_points)
    print('------')
    eval_result, dt_car_pos, nearest_positions, car_coordinates, vi_list = \
        gaussian_eval(gt_points, dt_points, value_only=False)
    print('Predicted car_coordinate', dt_car_pos)
    print('True car_coordinate', car_coordinates)
    print('nearest_pos', nearest_positions)
    print('vi_list', vi_list)
    print('eval_result', eval_result)

