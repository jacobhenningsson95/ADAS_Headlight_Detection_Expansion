import os
import pickle
import cv2
import torch.utils.data
import numpy as np
import torch
import json
import torch.utils.data


def resize_image(img, labels, new_size=512, max_num_car=8, max_num_light=2, test=False):
    """
    Resized image and keypoints. First the relevant keypoints are extracted from the labels.

    :param img: image to resize
    :param labels: Annotaions for this image
    :param new_size: Specified new size
    :param max_num_car: Max number of vehicles
    :param max_num_light: Max number of headlights per vehicle
    :param test: Boolean for test. During testing we need the position of the car and not the headlights.
    :return: resized image and scaled keypoints
    """
    original_img_size = img.shape
    new_img_size = (new_size, new_size)
    img_resized = cv2.resize(img, new_img_size)

    # Convert the resized image to a PyTorch Tensor
    img_resized = torch.from_numpy(img_resized.astype(np.float32) / 255.0)  # Normalize to [0, 1]

    # Calculate the scaling factors for X and Y dimensions
    x_scale = new_img_size[1] / original_img_size[1]
    y_scale = new_img_size[0] / original_img_size[0]
    keypoints = []  # save the keypoint
    # for eval method, we only need car pos
    if test:
        keypoint = []
        for annotation in labels['annotations']:
            keypoint.append([int(annotation['pos'][0] * x_scale), int(annotation['pos'][1] * y_scale), 1])

        while len(keypoint) < max_num_car:
             keypoint.append([0, 0, -1])
        keypoints.append(keypoint)
    # else for train
    else:
        for annotation in labels['annotations']:
            keypoint = []
            for instance in annotation['instances']:
                # Update the X and Y position values
                if instance["direct"] and not instance["rear"]:
                    keypoint.append([int(instance['pos'][0] * x_scale), int(instance['pos'][1] * y_scale), 1])
                if len(keypoint) == max_num_light:
                    break
            # make sure all data have same size
            while len(keypoint) < max_num_light:
                keypoint.append([0, 0, -1])
            keypoints.append(keypoint)

            if len(keypoints) == max_num_car:
                break

        while len(keypoints) < max_num_car:
            keypoint = []
            while len(keypoint) < max_num_light:
                keypoint.append([0, 0, -1])
            keypoints.append(keypoint)

    keypoints = torch.tensor(keypoints, dtype=torch.float32)
    if keypoints.numel() == 0:
        keypoints = torch.tensor([[[0., 0., -1.]]])

    return img_resized, keypoints


class GenerateSegmentationMap:
    """
    Initialize the Segmentation Map Generator.

    Args:
        max_n_instances (int): Maximum number of instances.
        output_res (int): Output resolution.
        num_parts (int): Number of parts.
        circle_radius (int): Circle radius.
        min_circle_radius (int): Minimum circle radius.
        max_circle_radius (int): Maximum circle radius.
        max_headlight_distance (int): Maximum headlight distance.
    """
    def __init__(self, max_n_instances=20, output_res=128, num_parts=2, circle_radius=2, min_circle_radius=2, max_circle_radius=25, max_headlight_distance=200):
        self.output_res = output_res
        self.num_parts = num_parts
        self.max_n_instances = max_n_instances
        self.circle_radius = circle_radius
        self.min_circle_radius = min_circle_radius
        self.max_circle_radius = max_circle_radius
        self.max_headlight_distance = max_headlight_distance
        self.background_color = (200, 200, 200)
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        self.car_radii = {}  # Dictionary to store circle radius for each car

    def __call__(self, keypoints, original_img=None):
        seg_map = np.full((self.output_res, self.output_res, 3), self.background_color, dtype=np.uint8)

        instance_seg = np.zeros(shape=(self.max_n_instances, self.output_res, self.output_res), dtype=np.uint8)

        img_height, img_width = original_img.shape[:2]

        instance_count = 0
        background_mask = np.zeros(shape=(self.output_res, self.output_res), dtype=np.uint8)

        for car_idx, kp_group in enumerate(keypoints):
            if car_idx >= self.num_parts:
                break

            circle_color = self.colors[car_idx % len(self.colors)]
            headlight_distance, car_distance, valid_headlights = self._calculate_distances(kp_group, img_width,
                                                                                           img_height)

            if car_idx not in self.car_radii:
                # Choose the first valid keypoint to calculate the radius
                for kp in kp_group:
                    if kp[2] > 0:
                        x, y = int(kp[0]), int(kp[1])
                        adjusted_radius = self._determine_circle_size(x, y, original_img, headlight_distance,
                                                                      car_distance, valid_headlights)
                        self.car_radii[car_idx] = adjusted_radius
                        break
            else:
                # Use the stored radius
                adjusted_radius = self.car_radii[car_idx]

            add_to_count = 0
            mask = np.zeros(shape=(self.output_res, self.output_res), dtype=np.uint8)
            for kp in kp_group:

                if kp[2] > 0:
                    x, y = int(kp[0]), int(kp[1])
                    # adjusted_radius = self._determine_circle_size(x, y, original_img, headlight_distance, car_distance)
                    mask_result = cv2.circle(mask, (x, y), adjusted_radius, 1, -1, lineType=cv2.LINE_AA)
                    background_mask = background_mask | mask_result
                    instance_seg[instance_count] = mask_result
                    cv2.circle(seg_map, (x, y), adjusted_radius, circle_color, -1, lineType=cv2.LINE_AA)
                    add_to_count = 1
            instance_count = instance_count + add_to_count

        instance_seg[instance_count] = np.invert(background_mask) / 255
        instance_count += 1


        semantic_seg = np.zeros(shape=(1, self.output_res, self.output_res), dtype=np.uint8)
        semantic_seg[0] = background_mask
        # semantic_seg[1] = np.invert(background_mask) / 255


        return seg_map, instance_seg, instance_count, semantic_seg

    def _calculate_distances(self, kp_group, img_width, img_height):
        ref_x = int(img_width * 0.6)
        ref_y = img_height - 1

        headlight_distances = []
        car_distances = []
        valid_headlights = 0

        for kp in kp_group:
            if kp[2] > 0:
                distance = np.linalg.norm(np.array([kp[0], kp[1]]) - np.array([ref_x, ref_y]))
                car_distances.append(distance)
                valid_headlights += 1

        if valid_headlights >= 2:
            headlight_distance = np.linalg.norm(np.array(kp_group[0][:2]) - np.array(kp_group[1][:2]))
            headlight_distances.append(headlight_distance)

        avg_headlight_distance = np.mean(headlight_distances) if headlight_distances else 0
        avg_car_distance = np.mean(car_distances) if car_distances else 0

        return avg_headlight_distance, avg_car_distance, valid_headlights

    def _determine_circle_size(self, x, y, original_img, headlight_distance, car_distance, valid_headlights):
        brightness_factor = self._measure_brightness(original_img, x, y) / 255

        if valid_headlights >= 2:
            # For cars with two headlights, use both headlight distance and brightness
            headlight_distance_factor = np.clip(headlight_distance / self.max_headlight_distance, 0, 1)
            combined_score = 1 * headlight_distance_factor #+ 0.1 * brightness_factor
        else:
            # For cars with one headlight, use only brightness and scale down
            combined_score = 0.5 * brightness_factor

        # Scale score to fit within min and max circle radius
        adjusted_radius = int(self.min_circle_radius + combined_score * (self.max_circle_radius - self.min_circle_radius))

        return adjusted_radius

    def _adjust_radius(self, brightness):
        # Normalize the brightness to a range between 0 and 1
        normalized_brightness = brightness / 255

        # Aggressive scaling
        adjusted_brightness = 1 - np.log(1 + (1 - normalized_brightness)) / np.log(2)

        # Calculate the adjusted radius
        return int(self.min_circle_radius + adjusted_brightness * (self.max_circle_radius - self.min_circle_radius))

    def _measure_brightness(self, img, x, y, window_size=10):
        # Ensure coordinates are within the bounds of the image
        x_min, x_max = max(0, x - window_size), min(img.shape[1], x + window_size)
        y_min, y_max = max(0, y - window_size), min(img.shape[0], y + window_size)

        # Extract the slice of the image around the keypoint
        slice = img[y_min:y_max, x_min:x_max]

        # Check if the slice is empty or contains NaN values
        if slice.size == 0 or np.isnan(slice).any():
            return 0  # Return a default low brightness value

        # Calculate and return the mean brightness
        return np.mean(slice)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, img_path, keypoint_path, size=512, max_num_car=8, max_num_light=2,
                 output_res=128, negative_samples=False, day_samples=False, test=False):
        """
        Custom dataset for loading the PVDN dataset and adapting the data for the stacked hourglass architecture.

        :param dataset_path: Path to the dataset.
        :param img_path: Image file directory.
        :param keypoint_path: Keypoints file directory.
        :param size: Image size that is input to the model.
        :param max_num_car: Maximum number of cars in each image.
        :param max_num_light: Maximum number of headlights in each image.
        :param output_res: Resolution of the output images.
        :param negative_samples: Flag indicating whether to use image containing no vehicles in the dataset.
        :param day_samples: Flag indicating whether to use images from the day dataset containing multiple vehicles.
        :param test: Flag indicating whether the dataset is for testing.
        """
        self.dataset_path = dataset_path
        self.img_path = img_path
        self.keypoint_path = keypoint_path
        self.negative_samples = negative_samples
        self.day_samples = day_samples
        self.ann_files_list = self.read_annotations()
        self.sequence_dir = self.read_sequence()
        self.size = size
        self.max_num_car = max_num_car
        self.max_num_light = max_num_light
        self.output_res = output_res
        self.generateSegmentationMap = GenerateSegmentationMap(output_res=self.output_res, num_parts=self.max_num_light)
        self.test = test
        self.start_index = 0
        self.end_index = self.__len__()


    def kpt_affine(self, keypoints):
        """
        Apply affine transformation to keypoints.
        """
        scale_factor = self.output_res / self.size
        scaled_keypoints = keypoints.clone()
        scaled_keypoints[:, :, :2] *= scale_factor
        return scaled_keypoints

    def read_annotations(self):
        """
        Filter and create dict containing paths to the annotations. If selected day dataset image will be added.
        :return: Dict containing paths to the annotations.
        """

        if not os.path.exists('Dataset Cache'):
            os.makedirs('Dataset Cache')

        if self.day_samples:
            cache_path = "Dataset Cache/" + self.img_path.split("/")[0]  + "_Day_Annoations.pkl"
        else:
            cache_path = "Dataset Cache/" + self.img_path.split("/")[0]  + "_Annoations.pkl"

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as file:
                return pickle.load(file)

        ann_files_list = []

        # Night data points that are direct lights and from the front are used in the dataset
        keypoint_dir = os.listdir( os.path.join(self.dataset_path, "night", self.keypoint_path))
        for file in keypoint_dir:
            path = os.path.join(self.dataset_path, "night", self.keypoint_path, file)
            with open(path, 'r') as json_file:
                labels = json.load(json_file)
            valid_label = self.negative_samples
            for annotation in labels['annotations']:
                if valid_label:
                    break
                for instance in annotation['instances']:
                    if valid_label:
                        break
                    if instance["direct"] and not instance["rear"]:
                        valid_label = True
            if valid_label:
                ann_files_list.append(("night", path))

        if self.day_samples:
            # Day data points that are direct lights, from the front and more than 2  are used in the dataset
            keypoint_dir = os.listdir(os.path.join(self.dataset_path, "day", self.keypoint_path))
            for file in keypoint_dir:
                path = os.path.join(self.dataset_path, "day", self.keypoint_path, file)
                with open(path, 'r') as json_file:
                    labels = json.load(json_file)
                nr_keypoints = 0
                for annotation in labels['annotations']:
                    for instance in annotation['instances']:
                        if instance["direct"] and not instance["rear"]:
                            nr_keypoints += 1
                if nr_keypoints>2:
                    ann_files_list.append(("day", path))

        with open(cache_path, 'wb') as file:
            pickle.dump(ann_files_list, file)

        return ann_files_list

    def generate_weights(self):
        """
        Assigns weights to images based on if they contain no, one ,or multiple vehicles.

        :return: List of weights corresponding to the images in the dataset.
        """
        if self.day_samples:
            cache_path = "Dataset Cache/" + self.img_path.split("/")[0]  + "_Day_Weights.pkl"
        else:
            cache_path = "Dataset Cache/" + self.img_path.split("/")[0]  + "_Weights.pkl"

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as file:
                return pickle.load(file)

        keypoint_dir = os.listdir(os.path.join(self.dataset_path, "night", self.keypoint_path))
        sample_dir = {}
        weights_dir = {}

        weights_dir["zero"] = 0
        weights_dir["one"] = 0
        weights_dir["multiple"] = 0

        for idx, file in enumerate(keypoint_dir):
            path = os.path.join(os.path.join(self.dataset_path, "night", self.keypoint_path), file)
            with open(path, 'r') as json_file:
                labels = json.load(json_file)
            nr_keypoints = 0
            for annotation in labels['annotations']:
                for instance in annotation['instances']:
                    if instance["direct"] and not instance["rear"]:
                        nr_keypoints += 1

            sample_dir[idx] = nr_keypoints

            if nr_keypoints == 0:
                weights_dir["zero"] += 1
            elif nr_keypoints == 1 or nr_keypoints == 2:
                weights_dir["one"] += 1
            else:
                weights_dir["multiple"] += 1

        """
        Since it's known that all day datapoints are multiple, add difference between the counted night weights
        and the len on the sample list to the nr of multiple samples.
        """
        if self.day_samples:
            weights_dir["multiple"] += (len(self.ann_files_list) - len(sample_dir))

        weights_dir["zero"] = 1 / weights_dir["zero"]
        weights_dir["one"] = 1 / weights_dir["one"]
        weights_dir["multiple"] = 1 / weights_dir["multiple"]

        sample_weights = []

        for i in range(len(sample_dir)):
            if sample_dir[i] == 0:
                sample_weights.append(weights_dir["zero"])
            elif sample_dir[i] == 1 or sample_dir[i] == 2:
                sample_weights.append(weights_dir["one"])
            else:
                sample_weights.append(weights_dir["multiple"])

        """
        Add samples weights for the day dataset samples. 
        """
        if self.day_samples:
            for i in range(len(sample_dir), len(self.ann_files_list)):
                sample_weights.append(weights_dir["multiple"])

        with open(cache_path, 'wb') as file:
            pickle.dump(sample_weights, file)

        return sample_weights

    def read_sequence(self):
        """
        The sequence json in the dataset contains information about which folder and image the annotation is associated
        with. This function reads that json file and stores the information in a dictionary.

        :return: Dict containing values that represent folders with corresponding keys based on image ids.
        """
        if self.day_samples:
            datasets = ["night", "day"]
            cache_path = "Dataset Cache/" + self.img_path.split("/")[0]  + "_Day_Sequence.pkl"
        else:
            datasets = ["night"]
            cache_path = "Dataset Cache/" + self.img_path.split("/")[0]  + "_Sequence.pkl"

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as file:
                return pickle.load(file)

        sequences_dir = {}
        for dataset in datasets:
            sequences_path = os.path.join(self.dataset_path, dataset, os.path.dirname(self.keypoint_path),
                                          'sequences.json')
            with open(sequences_path, 'r') as json_file:
                sequences = json.load(json_file)

                for sequence in sequences["sequences"]:
                    for img_id in sequence["image_ids"]:
                        sequences_dir[int(img_id)] = sequence["dir"]

        with open(cache_path, 'wb') as file:
            pickle.dump(sequences_dir, file)

        return sequences_dir

    def __len__(self):
        """len of dataset"""
        return int(len(self.ann_files_list) / 1)

    def preprocess(self, img_dir, labels_dir):
        """
        Load image and resize it.

        :param img_dir: path to the image
        :param labels_dir: path to the annotations
        :return: resized image and keypoints
        """
        # load img and keypoint
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        with open(labels_dir, 'r') as json_file:
            labels = json.load(json_file)
        # resize and read the keypoint
        img_resized, keypoints = resize_image(img, labels, new_size=self.size,
                                              max_num_car=self.max_num_car, max_num_light=self.max_num_light,
                                              test=self.test)
        """To do (add more preprocess)"""

        img_resized = img_resized[:, :, np.newaxis]

        keypoints = self.kpt_affine(keypoints)
        return img_resized, keypoints, img

    def generate_data(self, ann_path):
        """
        Finds to associated image to the annotation, loads it, resizes it and generates heatmaps based on the keypoints.

        :param ann_path: path to the annotation.
        :return: resized image, keypoints, image name. If the dataset is not for testing it will also return segmentation maps.
        instance maps, instance count and semantic_segmentations.
        """

        with open(ann_path[1], 'r') as json_file:
            annotation = json.load(json_file)

        if ann_path[0] == "night":
            img_path = os.path.join(self.dataset_path, "night", self.img_path,
                                    self.sequence_dir[annotation["image_id"]],
                                    os.path.basename(ann_path[1]).replace('.json', '.png'))
        elif ann_path[0] == "day":
            img_path = os.path.join(self.dataset_path, "day", self.img_path,
                                    self.sequence_dir[annotation["image_id"]],
                                    os.path.basename(ann_path[1]).replace('.json', '.png'))

        img_name = os.path.basename(img_path).replace('.png', '')
        img_name = img_name.lstrip('0')
        img_resized, keypoints, original_image = self.preprocess(img_path, ann_path[1])
        segmentation_maps, instance_seg, instance_count, semantic_seg = self.generateSegmentationMap(keypoints,
                                                                                                     original_image)

        if self.test:
            return img_resized, semantic_seg

        else:

            return img_resized, img_name, segmentation_maps, instance_seg, instance_count, semantic_seg

    def generate_data_by_name(self, img_name):
        """

        Based on an image name find the annotation and image path. This is used by the visualization code.

        :param img_name: Name of the image to return.
        :return: resized image, keypoints, image name, scaled keypoints for loss function and Heatmaps.
        """

        image_id = os.path.basename(img_name).replace('.png', '')
        image_id = image_id.lstrip('0')
        image_id = int(image_id)

        img_path = os.path.join(self.dataset_path, "night", self.img_path, self.sequence_dir[image_id], img_name)
        ann_path = os.path.join(self.dataset_path, "night", self.keypoint_path, os.path.basename(img_name).replace('.png', '.json'))

        img_resized, keypoints, original_image = self.preprocess(img_path, ann_path)
        segmentation_maps, instance_seg, instance_count, semantic_seg = self.generateSegmentationMap(keypoints, original_image)

        return img_resized, img_name, segmentation_maps, instance_seg, instance_count, semantic_seg

    def get_image_path(self, index):
        """
        Based on the index in the annotations dict find the image path. This is used by the video visualization code.

        :param index: Index of the image path.
        :return: image path of the associated index.
        """
        ann_path =  self.ann_files_list[index]

        with open(ann_path[1], 'r') as json_file:
            annotation = json.load(json_file)

        img_path = os.path.join(self.dataset_path, "night", self.img_path, self.sequence_dir[annotation["image_id"]], os.path.basename(ann_path[1]).replace('.json', '.png'))

        return img_path

    def find_sequence_indices(self, value_to_find):
        """
        If a specific sequence is selected for the video visualization this function sets the start_index and end_index
        to only include images of that sequence. By default the start_index is 0 and the end_index is the lenght
        of the dataset.

        :param value_to_find:
        """
        start_index = None
        end_index = None
        in_sequence = False

        for i, (key, value) in enumerate(self.sequence_dir.items()):
            if value == value_to_find:
                if not in_sequence:
                    start_index = i
                    in_sequence = True
                end_index = i
            elif in_sequence:
                break  # Exit the loop when the sequence ends

        self.start_index = start_index
        self.end_index = end_index

    def __getitem__(self, idx):
        return self.generate_data(self.ann_files_list[idx])