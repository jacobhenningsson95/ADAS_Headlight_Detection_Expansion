import warnings
import config as c
import torch
from dataset import Dataset
import argparse
import os
from train import load_checkpoint
from utils.vis_utils import initialize_visualizations, visualize_iteration
from utils.loss import calc_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_ordinal', default=0, type=int, metavar='N', help='Specify which graphics card to use')
    parser.add_argument('--model_filename', default="model_epoch0.pth", type=str, metavar='N', help='Filename of model to visualize')
    parser.add_argument('--image_name', default="002580.png", type=str, metavar='N', help='filename of image to visualize')
    parser.add_argument('--dataset_type', default="test", type=str, metavar='N', help='test/val or train')
    parser.add_argument('--simplified', default=False, type=bool, metavar='N', help='Simplified version of the visualization with less figures')
    args = parser.parse_args()
    return args


def main():
    config = c.__config__
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_ordinal)

    config, model, optimizer, start_epoch = load_checkpoint(args.model_filename, config, device)

    plot_dict = initialize_visualizations(args.simplified)
    criterion = calc_loss(config)

    model.eval()
    print('Model loaded successfully')
    # Create a test dataset instance
    test_dataset = Dataset(config["dataset_path"], args.dataset_type + "/images/", args.dataset_type +  "/labels/keypoints", max_num_car=config['max_num_car'],
                           max_num_light=config['max_num_light'], output_res=config['output_res'], negative_samples=True)
    # Generate data for the specified image
    img, img_name, seg_maps, instance_maps, instance_count, semantic_seg = test_dataset.generate_data_by_name(args.image_name)
    # Convert data to PyTorch tensors and move to the specified device
    img = torch.Tensor(img)
    seg_maps = torch.Tensor(seg_maps)
    instance_maps = torch.Tensor(instance_maps)
    instance_count = torch.Tensor([instance_count]).to(torch.uint8)
    semantic_seg = torch.Tensor(semantic_seg)

    img = img.to(device)
    seg_maps = seg_maps.to(device)
    instance_maps = instance_maps.to(device)  # 16 20 128 128
    instance_count = instance_count.to(device)
    semantic_seg = semantic_seg.to(device)
    img = img[None,:,:,:]
    seg_maps = seg_maps[None,:,:,:]
    instance_maps = instance_maps[None,:,:,:]
    semantic_seg = semantic_seg[None,:,:,:]

    output = model(img)
    # Calculate the loss
    result = criterion(output, instance_maps=instance_maps, instance_count=instance_count, semantic_seg=semantic_seg)

    loss = 0
    for i in result:
        loss = loss + torch.mean(i)
    # Visualize the iteration using the visualization functions
    visualize_iteration(output, img, instance_maps, loss, plot_dict, semantic_seg=semantic_seg, seg_maps=seg_maps, instance_count=instance_count, simplified_vis=args.simplified)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()