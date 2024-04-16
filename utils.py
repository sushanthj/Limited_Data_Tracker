import numpy as np
import torch
import cv2
import glob
import os
import imageio
from torchvision.transforms import v2
from PIL import Image
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_square(image):
    h, w, _ = image.shape
    desired_size = max(h,w)
    if w > h:
        padding_y = max(0, (desired_size - h) // 2)

        # Create a new square canvas filled with zeros (black)
        square_image = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)

        # Place the original image in the center
        square_image[padding_y:padding_y + h, :, :] = image
    else:
        padding_x = max(0, (desired_size - w) // 2)

        # Create a new square canvas filled with zeros (black)
        square_image = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)

        # Place the original image in the center
        square_image[:, padding_x:padding_x+w, :] = image

    return square_image

def get_crops(masks, image):
    # masks = list of dictionaries with keys 'bbox' which has mask in XYWH format
    # crop according to each mask and store it in a list
    cropped_images = []
    for mask in masks:
        x, y, w, h = mask['bbox']

        if (w*h < 700):
            continue

        maxlen = max(w,h)

        centroid_x = x + w/2
        centroid_y = y + h/2

        ymin = min(max(0,int(centroid_y - maxlen)), image.shape[0])
        ymax = min(max(0,int(centroid_y + maxlen)), image.shape[0])
        xmin = min(max(0,int(centroid_x - maxlen)), image.shape[0])
        xmax = min(max(0,int(centroid_x + maxlen)), image.shape[0])

        # check if nothing got cropped
        if (ymin + ymax == h) and (xmin + xmax == w):
            cropped_images.append(image[ymin:ymax, xmin:xmax, :])
        # if some part of the image did get cropped, pad it
        else:
            cropped_images.append(make_square(image[ymin:ymax, xmin:xmax, :]))

    return cropped_images

def normalize_bbox(bbox, image_shape):
    height, width = image_shape[:2]
    xmin, ymin, w, h = bbox
    # Normalize bounding box by image size
    xmin /= width
    ymin /= height
    w /= width
    h /= height
    return np.array([xmin, ymin, w, h], dtype=float)

def get_crop_and_masks_per_image(image, mask_generator, args):
    image = image.squeeze(0) #TODO : Add batch support
    image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0) # H,W,C
    image = (image * 255).astype(np.uint8)

    masks = mask_generator.generate(image)
    crops = get_crops(masks, image)
    
    # transform the crops to be able to give to classification model
    # channel_wise_means, channel_wise_stds = get_mean_and_std_calculated(args.data_dir)
    # Means and standard dev found using disabled function above
    channel_wise_means = [0.485, 0.456, 0.406]
    channel_wise_stds = [0.229, 0.224, 0.225]
    classification_transforms = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=channel_wise_means, std=channel_wise_stds),
    ])

    crops = [classification_transforms(Image.fromarray(crop)) for crop in crops]
    # batch crops (N, C, H, W), N = number of crops
    if len(crops) > 0:
        batch_crops = torch.stack(crops).to(DEVICE)
        return batch_crops, masks
    else:
        return None, masks

"""
# Without DeepSort Tracking
def process_outputs(image, masks, preds, save_dir, index, tracker):
    # use preds to find index of true masks
    # normalize image to save in cv2
    image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    # conver to uint8
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    true_mask_indices = np.where(preds == 1)[0]    
    
    if len(true_mask_indices) > 0:
        for i in true_mask_indices:
            x, y, w, h = [int(z) for z in masks[i]['bbox']]
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # save image
    save_path = os.path.join(save_dir, f"output_{index}.png")
    cv2.imwrite(save_path, image)
"""

def tlwh_to_xywh(bbox_tlwh):
    xmin, ymin, w, h = bbox_tlwh
    # Calculate the center coordinates
    x = xmin + w / 2
    y = ymin + h / 2
    return [x, y, w, h]

SKIP_FRAMES = 1

def convert_images_to_gif(data_dir, output_path):
    img_paths = sorted(glob.glob(data_dir + "/*.png"))
    images = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    imageio.mimsave(
        output_path,
        [np.array(img) for i, img in enumerate(images) if i % SKIP_FRAMES == 0],
        duration=0.1,
        loop=0x7FFF * 2 + 1,
    )
    
def make_dirs(inp_path):
    if not os.path.exists(inp_path):
        os.makedirs(inp_path)