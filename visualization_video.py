import warnings
import config as c
import torch
from tqdm import tqdm
from dataset import Dataset
import argparse
import os
from train import load_checkpoint
from utils.vis_utils import cluster, count_blobs_in_segmentation_map
import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
Save imgs as Video
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_ordinal', default=0, type=int, metavar='N', help='Specify which graphics card to use')
    parser.add_argument('--model_filename', default="model_epoch0.pth", type=str, metavar='N', help='Filename of model to visualize')
    parser.add_argument('--sequence_folder', default=None, type=str, metavar='N', help='specific folder sequence')
    parser.add_argument('--dataset_type', default="test", type=str, metavar='N', help='test/val or train')
    parser.add_argument('--raw_instance', default=False, type=bool, metavar='N', help='Get the raw instances without keypoint prediction')
    args = parser.parse_args()
    return args

def main():
    config = c.__config__
    args = parse_args()

    if not os.path.exists('Video Output'):
        os.makedirs('Video Output')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_ordinal)

    config, model, optimizer, start_epoch = load_checkpoint(args.model_filename, config, device)

    model.eval()
    print('Model loaded successfully')

    print("Loading dataset...")
    test_dataset = Dataset(config["dataset_path"], args.dataset_type + "/images", args.dataset_type +  "/labels/keypoints", max_num_car=config['max_num_car'],
                           max_num_light=config['max_num_light'], output_res=config['output_res'], negative_samples=True)
    print("Dataset loaded!")

    if args.sequence_folder != None:
        test_dataset.find_sequence_indices(args.sequence_folder)
        output_video_path = os.path.join("Video Output", args.model_filename.split("/")[-1] + "_" + args.sequence_folder + '_output_video.mp4')
    else:
        output_video_path = os.path.join("Video Output", args.model_filename.split("/")[-1] + "_" + args.dataset_type + '_output_video.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(output_video_path)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 24, (1280, 960))

    if args.sequence_folder != None:
        test_dataset.find_sequence_indices(args.sequence_folder)

    video_colors = [
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 128, 0),  # Olive
        (0, 128, 128)  # Teal
    ]

    for i in tqdm(range(test_dataset.start_index, test_dataset.end_index)):
        img_cuda, img_name, seg_maps, instance_maps, instance_count, semantic_seg = test_dataset.__getitem__(i)

        img_path = test_dataset.get_image_path(i)

        img = cv2.imread(img_path)
        img_cuda = torch.Tensor(img_cuda)

        img_cuda = img_cuda.to(device)

        img_cuda = img_cuda[None, :, :, :]

        output = model(img_cuda)

        instance_output = torch.mean(output[0], dim=1)
        semantic_output = torch.mean(output[1], dim=1)


        sem_seg_prediction, ins_seg_prediction, embeddings_comp = cluster(semantic_output[0], instance_output[0])

        if args.raw_instance:
            ins_seg_prediction = cv2.resize(ins_seg_prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        _n_clusters = len(np.unique(ins_seg_prediction.flatten())) - 1 # discard bg
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]
        ins_seg_pred_color = np.zeros((ins_seg_prediction.shape[0], ins_seg_prediction.shape[1], 3), dtype=np.uint8)
        for i in range(_n_clusters):

            if args.raw_instance:
                img[ins_seg_prediction == (i + 1)] = (np.array(colors[i][:3]) * 255).astype('int')
            else:
                ins_seg_pred_color[ins_seg_prediction == (i + 1)] = (np.array(colors[i][:3]) * 255).astype('int')

        if not args.raw_instance:
            point_pairs, processed_instance_output = count_blobs_in_segmentation_map(ins_seg_pred_color)

            vehicle_coordinates = []

            for idx, (vehicle, vehicle_lights) in enumerate(point_pairs.items()):
                vehicle_coordinates.append( list(vehicle_lights))

            if len(vehicle_coordinates) != 0:
                vehicle_coordinates = sorted(vehicle_coordinates, key=lambda x: x[0][0])
                for i, coordinates in enumerate(vehicle_coordinates):
                    for coordinate in coordinates:
                        if coordinate != None:
                            try:
                                cv2.circle(img, (int(coordinate[0]*4/0.4), int(coordinate[1]*4/0.53333)) ,7, video_colors[i], -1)
                            except NameError:
                                print(coordinate)

        video_writer.write(img)
    video_writer.release()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()