import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
import ipdb
from torchvision import models, transforms
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from classification.dataloader import get_test_dataloaders
from classification.utils import get_trained_model
from classification.dataloader import get_mean_and_std_calculated
from utils import make_square, get_crops, get_crop_and_masks_per_image
from utils import tlwh_to_xywh, make_dirs, convert_images_to_gif
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything import SamPredictor
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

sys.path.append("..")

config = {
          'num_classes': 2,
          'lr': 1e-4,
          'sam_checkpoint' : "sam_vit_h_4b8939.pth",
          'model_type' : "vit_h",
        #   'init_bbox' : [6.00,166.00, 43.00, 27.0], # tlwh
        }


def load_models(args):
    # Load Classification Model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    # 2 = num output classes (here Car or Not Car)
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # load model from checkpoint
    model, _, _ = get_trained_model(model, args.checkpoint_path, scheduler, optimizer)

    # Load SAM Model
    sam = sam_model_registry[config['model_type']](checkpoint=config['sam_checkpoint'])
    sam.to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.98,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=600,  # Requires open-cv to run post-processing
    )

    return model, mask_generator


def process_outputs(image, masks, preds, save_dir, index, tracker):
    # use preds to find index of true masks
    # normalize image to save in cv2
    image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    # conver to uint8
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    true_mask_indices = np.where(preds == 1)[0]
    
    if len(true_mask_indices) > 0:
        bboxes = []
        for i in true_mask_indices:
            xmin, ymin, w, h = [int(z) for z in masks[i]['bbox']] # tlwh
            bboxes.append(np.array(tlwh_to_xywh((xmin, ymin, w, h)))) # Tracker requires xywh

        bboxes_np = np.vstack(bboxes, dtype=float)
        confs = np.ones(bboxes_np.shape[0])
        tracks = tracker.update(bboxes_np, confs, image)
    else:
        tracker.forward_predict()
    
    for track in tracker.tracker.tracks:
        hits = track.hits
        x1, y1, w, h = track.to_tlwh()
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        image = cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)

    # save image
    save_path = os.path.join(save_dir, f"output_{index}.png")
    cv2.imwrite(save_path, image)


def inference(args):
    data_dir = os.path.join(args.data_dir)
    save_dir = os.path.join(args.save_dir, 'images')
    gif_dir = os.path.join(args.save_dir, 'gifs')
    make_dirs(save_dir)
    make_dirs(gif_dir)
    data_loader = get_test_dataloaders(data_dir, batch_size=1) # TODO: Add Batch Support

    # Load SAM and Classification Model
    model, mask_generator = load_models(args)
    # Load DeepSort
    kalman_tracker = DeepSort(model_path=args.deepsort_path, max_age=3)

    model.eval()
    batch_bar = tqdm(total=len(data_loader),
                     dynamic_ncols=True,
                     position=0, leave=False, desc='Test')

    for i, (images) in enumerate(data_loader):
        # loads one image at a time (C, H, W) # TODO: Add Batch Support
        images = images.to(DEVICE)

        # Run SAM and get crops (N, C, H, W), N = number of crops per image
        crops, masks = get_crop_and_masks_per_image(images, mask_generator, args)

        if crops is not None:
            with torch.inference_mode():
                outputs = model(crops)

            outputs = torch.argmax(outputs, axis=-1).detach().cpu().numpy()
            # class 1 = car
        else:
            outputs = np.zeros(len(masks)) # return all false predictions

        # save outputs in args.save_dir
        process_outputs(images, masks, outputs, save_dir, i, kalman_tracker)

        batch_bar.update()

    batch_bar.close()
    convert_images_to_gif(save_dir, os.path.join(gif_dir, 'output.gif'))
    print(f"Done!")
    print(f"Images saved to {save_dir} and GIF saved to {gif_dir}")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_path', type=str, default='./classification/checkpoints/checkpoint.pth')
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--deepsort_path', type=str, default='./deep_sort/deep/checkpoint/ckpt.t7')
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # Do Inference!
    inference(args)
    
if __name__ == "__main__":
    main()