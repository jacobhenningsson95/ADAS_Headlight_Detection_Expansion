import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans
from scipy.ndimage import label
import cv2
from sklearn.manifold import TSNE


def count_blobs_in_segmentation_map(seg_map, kmeans_threshold_ratio=1.0):
    """
    Count blobs in the segmentation map and return modified segmentation map with extracted keypoints.

    Args:
        seg_map (np.ndarray): Segmentation map.
        kmeans_threshold_ratio (float): Threshold ratio for K-means clustering.

    Returns:
        tuple: Tuple containing point pairs and modified segmentation map.

    """
    unique_colors, counts = np.unique(seg_map.reshape(-1, seg_map.shape[2]), axis=0, return_counts=True)
    most_frequent_color = unique_colors[np.argmax(counts)]
    modified_seg_map = seg_map.copy()

    point_pairs = {}  # Dictionary to store point pairs for each color

    for color in unique_colors:
        if np.array_equal(color, most_frequent_color):
            continue

        mask = np.all(seg_map == color, axis=-1)
        labeled_array, num_features = label(mask)

        for feature in range(1, num_features + 1):
            feature_mask = labeled_array == feature
            y, x = np.where(feature_mask)
            blob_width = np.max(x) - np.min(x)
            blob_height = np.max(y) - np.min(y)

            if blob_width <= kmeans_threshold_ratio * blob_height:
                # Use shape analysis
                center_x, center_y = int(np.mean(x)), int(np.mean(y))
                modified_seg_map[center_y, center_x] = [255, 255, 255]
                point_pairs.setdefault(tuple(color), []).append((center_x, center_y))
            else:
                # Use K-means clustering
                points = np.column_stack((x, y))
                kmeans = KMeans(n_clusters=2).fit(points)
                centroids = kmeans.cluster_centers_.astype(int)
                for centroid in centroids:
                    cx, cy = centroid
                    modified_seg_map[cy, cx] = [255, 255, 255]
                    point_pairs.setdefault(tuple(color), []).append((cx, cy))

    return point_pairs, modified_seg_map


def upsample_prediction(prediction, image_height, image_width):
    """
    Upsample the prediction to match the original image dimensions.
    Args:
        prediction (np.ndarray): Prediction map.
        image_height (int): Original image height.
        image_width (int): Original image width.
    """
    return cv2.resize(prediction, (image_width, image_height),
                      interpolation=cv2.INTER_NEAREST)


def cluster(sem_seg_prediction, ins_seg_prediction):
    """
    Cluster the semantic and instance segmentation predictions.

    Args:
        sem_seg_prediction (torch.Tensor): Semantic segmentation prediction.
        ins_seg_prediction (torch.Tensor): Instance segmentation prediction.

    Returns:
        tuple: Tuple containing semantic segmentation threshold, instance segmentation map, and embeddings.
    """
    sem_seg_prediction = torch.sigmoid(sem_seg_prediction)

    sem_seg_prediction = sem_seg_prediction  # 1, 128,128
    ins_seg_prediction = ins_seg_prediction

    seg_height, seg_width = ins_seg_prediction.shape[1:]

    sem_seg_prediction = sem_seg_prediction.cpu().detach().numpy()
    sem_seg_prediction_thresh = (sem_seg_prediction > 0.5).squeeze().astype(np.uint8)

    embeddings_comp = []
    if  0 != np.max(sem_seg_prediction_thresh):

        embeddings = ins_seg_prediction.cpu()
        embeddings = embeddings.detach().numpy()
        embeddings = embeddings.transpose(1, 2, 0)  # h, w, c 128,128,32

        embeddings_comp = []
        for i in range(embeddings.shape[2]):
            result = embeddings[:, :, i][sem_seg_prediction_thresh  != 0]
            if len(result) != 0:
                embeddings_comp.append(result)

        embeddings_comp = np.stack(embeddings_comp, axis=1) # 128, 32, 128
        labels = DBSCAN(n_jobs=-1, eps=0.85, min_samples=2).fit_predict(embeddings_comp)

    instance_mask = np.zeros((seg_height, seg_width), dtype=np.uint8)

    fg_coords = np.where(sem_seg_prediction_thresh != 0)
    for si in range(len(fg_coords[0])):
        y_coord = fg_coords[0][si]
        x_coord = fg_coords[1][si]
        _label = labels[si] + 1
        instance_mask[y_coord, x_coord] = _label

    return sem_seg_prediction_thresh, instance_mask,  embeddings_comp


def visualize_iteration(output, img, instance_maps, loss, plot_dict, semantic_seg, seg_maps, instance_count, simplified_vis=False):
    """
    Uses the initialized plot dict to visualize output from the model and ground truths.

    :param output: Raw output of the model
    :param img: Original image
    :param keypoints: True keypoint
    :param heatmaps: True heatmaps
    :param loss: Loss value for the current output
    :param config: Configuration for model
    :param plot_dict: Initialized plot dict
    """
    instance_output = torch.mean(output[0], dim=1)
    semantic_output = torch.mean(output[1], dim=1)

    sem_seg_prediction, ins_seg_prediction, embeddings_comp = cluster(semantic_output[0],
                                            instance_output[0])


    semantic_sigmoid = torch.sigmoid(semantic_output[0].squeeze())
    count_above_threshold = (semantic_sigmoid > 0.5).sum().item()



    _n_clusters = len(np.unique(ins_seg_prediction.flatten()))-1 # discard bg
    colors = [plt.cm.Paired(each) for each in np.linspace(0, 1, _n_clusters+1)]
    ins_seg_pred_color = np.zeros((ins_seg_prediction.shape[0], ins_seg_prediction.shape[1], 3), dtype=np.uint8)

    label_colors = []
    reduced_list = [None,None]

    if count_above_threshold != 0:
        for i in range(_n_clusters):
            ins_seg_pred_color[ins_seg_prediction == (
                    i + 1)] = (np.array(colors[i][:3]) * 255).astype('int')

        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)

        random_indices = np.random.choice(range(count_above_threshold ), size=400)

        embeddings_comp_choice = embeddings_comp[random_indices]
        reduced_embeddings = tsne.fit_transform(embeddings_comp_choice)
        labels_choice = np.stack([ins_seg_prediction[np.where(sem_seg_prediction == 1)]],axis=1)
        labels_choice = labels_choice[random_indices]
        labels_choice = labels_choice.flatten()

        label_colors = []
        for idx, l in enumerate(labels_choice):
            if l == _n_clusters:
                label_colors.append(np.array([0, 0, 0]))
            else:
                label_colors.append(np.array(colors[l][:3]))

        label_colors = np.array(label_colors)
        reduced_list = list(zip(reduced_embeddings[:,0], reduced_embeddings[:,1]))

    # Using count_blobs_in_segmentation_map on the instance prediction
    point_pairs, processed_instance_output = count_blobs_in_segmentation_map(ins_seg_pred_color)


    headlight_keypoints = []
    headlight_keypoint_color = []
    for idx, (vehicle, vehicle_lights) in enumerate(point_pairs.items()):
        for light in vehicle_lights:
            headlight_keypoints.append(light[1]*4)
            headlight_keypoints.append(light[0]*4)
            headlight_keypoint_color.append(np.array(colors[idx][:3]))

    if simplified_vis:

        plot_dict["img"].set_data(((img[0] * 255).cpu().detach().numpy()))
        plot_dict["img_plot"].set_data(((img[0] * 255).cpu().detach().numpy()))
        plot_dict["instance_prediction"].set_data(ins_seg_pred_color)
        plot_dict["embedding_space"].set_offsets(reduced_list)
        plot_dict["embedding_space"].set_color(label_colors)

        if len(headlight_keypoints) != 0:
            plot_dict["img_det_keypoints"].set_offsets(list(zip(headlight_keypoints[1::2], headlight_keypoints[::2])))
            plot_dict["img_det_keypoints"].set_color(headlight_keypoint_color)
        else:
            plot_dict["img_det_keypoints"].set_offsets(np.empty((0, 2)))

    else:
        plot_dict["img"].set_data(((img[0] * 255).cpu().detach().numpy()))
        plot_dict["img_plot"].set_data(((img[0] * 255).cpu().detach().numpy()))

        if len(headlight_keypoints) != 0:
            plot_dict["img_det_keypoints"].set_offsets(list(zip(headlight_keypoints[1::2], headlight_keypoints[::2])))
            plot_dict["img_det_keypoints"].set_color(headlight_keypoint_color)
        else:
            plot_dict["img_det_keypoints"].set_offsets(np.empty((0, 2)))

        plot_dict["target_seg"].set_data(seg_maps[0].squeeze().cpu().numpy())
        plot_dict["target_semantic"].set_data((semantic_seg[0][0]*255).cpu().detach().numpy())
        plot_dict["first_instance"].set_data(instance_maps[0][0].squeeze().cpu().numpy()*255)
        plot_dict["semantic_pred"].set_data(semantic_sigmoid.cpu().detach().numpy()*255)
        plot_dict["instance_prediction"].set_data(ins_seg_pred_color)
        plot_dict["embedding_space"].set_offsets(reduced_list)
        plot_dict["embedding_space"].set_color(label_colors)


    plot_dict["fig"].suptitle("Predicted instance count: " + str(_n_clusters) + ", True instance count: " + str(int(instance_count[0])-1) + ", Loss: " + str(round(loss.item(),3)) )
    plot_dict["fig"].canvas.draw()
    plot_dict["fig"].canvas.flush_events()
    plt.pause(0.1)


def initialize_visualizations(simplified_vis=False):
    """
    Initializes a dict for visualization.

    :param config: Configuration for the current model
    :return: Initialized visualization dict.
    """
    plt.ion()

    plot_dict = {}

    if simplified_vis:
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        plot_dict["fig"] = fig
        plt.subplots_adjust(hspace=0.4)
        plt.subplots_adjust(right=0.945)

        plot_dict["img"] = axes[0].imshow(np.zeros((512, 512)), cmap='gray', vmin=0, vmax=255)
        axes[0].set_title("Image")

        plot_dict["instance_prediction"] = axes[1].imshow(np.zeros((128, 128)), vmin=0, vmax=255)
        axes[1].set_title("Instance prediction")

        plot_dict["embedding_space"] = axes[2].scatter(None, None)
        axes[2].set_xlim(-600, 600)
        axes[2].set_ylim(-600, 600)
        axes[2].set_title("Embedding space")

        plot_dict["img_plot"] = axes[3].imshow(np.zeros((512, 512)), cmap='gray', vmin=0, vmax=255)
        plot_dict["img_det_keypoints"] = axes[3].scatter(None, None, color="blue")
        axes[3].set_title("Extracted keypoints")

    else:

        fig, axes = plt.subplots(2, 4, figsize=(14, 4))
        plot_dict["fig"] = fig
        plt.subplots_adjust(hspace=0.4)
        plt.subplots_adjust(right=0.945)

        plot_dict["img"] = axes[0][0].imshow(np.zeros((512, 512)), cmap='gray', vmin=0, vmax=255)
        axes[0][0].set_title("Image")

        plot_dict["target_seg"] = axes[1][0].imshow(np.zeros((128, 128, 3)), vmin=0, vmax=255)
        axes[1][0].set_title("Target instance Map")

        plot_dict["target_semantic"] = axes[0][1].imshow(np.zeros((128, 128, 3)), vmin=0, vmax=255)
        axes[0][1].set_title("Target Semantic Map")

        plot_dict["first_instance"] = axes[1][1].imshow(np.zeros((128, 128, 3)), vmin=0, vmax=255)
        axes[1][1].set_title("Target first Instance Map")

        plot_dict["semantic_pred"] = axes[0][2].imshow(np.zeros((128, 128, 3)), vmin=0, vmax=255)
        axes[0][2].set_title("Semantic Prediction")

        plot_dict["instance_prediction"] = axes[1][2].imshow(np.zeros((128, 128)), vmin=0, vmax=255)
        axes[1][2].set_title("Instance prediction")

        plot_dict["embedding_space"] = axes[0][3].scatter(None, None)
        axes[0][3].set_xlim(-600, 600)
        axes[0][3].set_ylim(-600, 600)
        axes[0][3].set_title("Embedding space")

        plot_dict["img_plot"] = axes[1][3].imshow(np.zeros((512, 512)), cmap='gray', vmin=0, vmax=255)
        plot_dict["img_det_keypoints"] = axes[1][3].scatter(None, None, color="blue")
        axes[1][3].set_title("Extracted keypoints")


    return plot_dict