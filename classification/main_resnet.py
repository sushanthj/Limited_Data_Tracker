import torch
import torchvision
import os
import gc
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
import pandas as pd
import torch.nn.init as init
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser
import glob
import wandb
import matplotlib.pyplot as plt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
print("Device: ", DEVICE)

from dataloader import get_train_dataloaders, get_test_dataloaders
from model import Network
from utils import train_model, validate_model, get_trained_model
import os
import torchvision.models as models

config = {
        'batch_size': 32,
        'lr': 1e-3,
        'epochs': 100,
        'num_classes': 2,
        'stochastic_depth_prob': 0.3,
        'truncated_normal_mean' : 0,
        'truncated_normal_std' : 0.2,
        }

def train(args):
    data_dir    = args.data_dir
    train_dir   = os.path.join(data_dir, "train")
    val_dir     = os.path.join(data_dir, "dev")
    test_dir    = os.path.join(data_dir, "test")

    train_loader, valid_loader, train_dataset = get_train_dataloaders(train_dir,
                                                                      val_dir,
                                                                      config)

    print("Number of classes    : ", len(train_dataset.classes))
    print("No. of train images  : ", train_dataset.__len__())
    print("Shape of image       : ", train_dataset[0][0].shape)
    print("Batch size           : ", config['batch_size'])
    print("Train batches        : ", train_loader.__len__())
    print("Val batches          : ", valid_loader.__len__())

    # Initialize the model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # 2 = num output classes (here Car or Not Car)
    model.fc = torch.nn.Linear(num_ftrs, 2)

    # freeze all layers except the FC layer
    for param in model.parameters():
        param.requires_grad = False

    # Set requires_grad to True for the last few layers
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    # Multi-Class Classification
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=0.05)
    gamma = 0.6
    milestones = [3,6,8,10,12,15]
    # scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.9, total_iters=5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    scaler = torch.cuda.amp.GradScaler()
    best_valacc = 0.0
    checkpoint_path = args.checkpoint_path

    for epoch in range(config['epochs']):

        curr_lr = float(optimizer.param_groups[0]['lr'])

        train_acc, train_loss = train_model(model, train_loader, optimizer,
                                            criterion, scheduler, config, DEVICE, scaler)

        print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1,
            config['epochs'],
            train_acc,
            train_loss,
            curr_lr))

        val_acc, val_loss = validate_model(model, valid_loader, criterion, config, DEVICE)

        print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))

        # #Save model in drive location if val_acc is better than best recorded val_acc
        if val_acc >= best_valacc:
            #path = os.path.join(root, model_directory, 'checkpoint' + '.pth')
            print("Saving model")
            # save locally
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch}, './checkpoint.pth')
            # save in drive as well
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch}, checkpoint_path)
            best_valacc = val_acc

        scheduler.step()

    # End of training
    gc.collect() # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()


def test(args):
    data_dir    = args.data_dir
    test_dir    = os.path.join(data_dir, "test")
    test_loader = get_test_dataloaders(test_dir, config)

    # Initialize the model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    # 2 = num output classes (here Car or Not Car)
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=0.05)
    gamma = 0.6
    milestones = [10,20,40,60,80]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.9, total_iters=5)

    # load model from checkpoint
    model, optimizer, scheduler = get_trained_model(model, args.checkpoint_path, scheduler, optimizer)

    model.eval()
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
    test_results = []

    for i, (images) in enumerate(test_loader):
        images = images.to(DEVICE)

        with torch.inference_mode():
            outputs = model(images)

        outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()

        test_results.extend(outputs)

        batch_bar.update()

    batch_bar.close()

    return test_results

def inference():
    args = parse_args()
    test_results = test(args)
    return test_results


def infer_single_image(img):
    img = Image.fromarray(img)
    channel_wise_means = [0.485, 0.456, 0.406]
    channel_wise_stds = [0.229, 0.224, 0.225]
    test_transforms = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=channel_wise_means, std=channel_wise_stds),
    ])
    # test_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((224, 224)),
    #     transforms.Normalize(mean=channel_wise_means, std=channel_wise_stds)
    # ])

    img = test_transforms(img)
    print(img.shape)

    # Initialize the model
    # model = models.resnet18(pretrained=True)
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    # 2 = num output classes (here Car or Not Car)
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.9, total_iters=5)

    # load model from checkpoint
    model, optimizer, scheduler = get_trained_model(model, './checkpoints/checkpoint.pth', scheduler, optimizer)

    model.eval()

    with torch.inference_mode():
        outputs = model(img.unsqueeze(0).to(DEVICE))
    outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy()

    return outputs


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='/projects/academic/rohini/m44/git-prjs/3DVision/Mach/classification/data')
    parser.add_argument('--stochastic_depth_prob', type=float, default=0.3)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/checkpoint.pth')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_classes', type=int, default=2)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Config
    config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'num_classes': args.num_classes,
        'stochastic_depth_prob': args.stochastic_depth_prob,
        'truncated_normal_mean' : 0,
        'truncated_normal_std' : 0.2,
    }
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    if args.mode == 'train':
        train(args)
    else:
        test(args)