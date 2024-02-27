import config as c
from torch.cuda.amp import autocast
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model import PoseNet
from tqdm import tqdm
from dataset import Dataset
import argparse
import warnings
import os
import matplotlib

from train import load_checkpoint
from utils.vis_utils import cluster


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_ordinal', default=0, type=int, metavar='N', help='Specify which graphics card to use')
    parser.add_argument('--cuda', default=True, type=int, metavar='N', help='Specify if cuda')
    parser.add_argument('--visualizations', default=False, type=bool, metavar='N',
                        help='Enables visualisations in the training loop')
    parser.add_argument('--model_path', default="model_epoch1.pth", type=str, metavar='N',
                        help='model path')
    args = parser.parse_args()
    return args


def data_gen(config):
    test_dataset = Dataset(config["dataset_path"], "test/images", "test/labels/keypoints",
                           size=config['input_res'],
                           max_num_car=config['max_num_car'],
                           max_num_light=config['max_num_light'],
                           output_res=config['output_res'],
                           negative_samples=True, day_samples=False, test=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size']*2,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True)

    return test_loader


def test_model(config, model, device):

    test_loader = data_gen(config)
    model.eval()
    iou_score = 0
    counting = 0
    with torch.no_grad():
        with autocast(enabled=config['autocast']):
            for img, semantic_seg in tqdm(test_loader):
                counting += 1
                img = img.to(device)
                output = model(img)  # torch.Size([batch_size, n_stack, output_dim, 128, 128])
                instance_output = torch.mean(output[0], dim=1)
                semantic_output = torch.mean(output[1], dim=1)
                sem_seg_prediction, ins_seg_prediction, embeddings_comp = cluster(semantic_output[0],
                                                                                  instance_output[0])
                _n_clusters = len(np.unique(ins_seg_prediction.flatten())) - 1  # discard bg
                semantic_sigmoid = torch.sigmoid(semantic_output[0].squeeze())
                semantic_sigmoid = semantic_sigmoid.cpu().detach().numpy()

                Target = semantic_seg.cpu().detach().numpy()[0][0]
                pre_semantic = (semantic_sigmoid >= 0.5).astype(int)
                intersection = np.logical_and(Target, pre_semantic).sum()
                union = np.logical_or(Target, pre_semantic).sum()
                # compute IoU
                if Target.sum() == 0 and pre_semantic.sum() == 0:
                    iou = 1
                elif np.sum(union) != 0:
                    iou = np.sum(intersection) / np.sum(union)
                else:
                    iou = 0
                iou_score += iou
    return iou_score/counting


def seed_and_settings(seed_value=46, args=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = c.__config__
    warnings.filterwarnings("ignore")
    matplotlib.use("TkAgg")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # seed everything
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_ordinal)
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    return device, config


def model_load(config, args, device):
    # create and load model
    poseNet = PoseNet(config['nstack'], config['inp_dim'], config['oup_dim'], bn=config['bn'],
                      increase=config['increase'])
    model = poseNet.to(device)
    model_dic = torch.load("Trained model/" + args.model_path, map_location=torch.device('cuda'))
    model.load_state_dict(model_dic['model'])
    print('Model path:', args.model_path)
    return model


def main():
    args = parse_args()
    device, config = seed_and_settings(args=args)
    config, model, optimizer, start_epoch = load_checkpoint(args.model_path, config, device)
    eval_value = test_model(config, model, device)
    print('Model score is:', eval_value)


if __name__ == '__main__':
    main()
