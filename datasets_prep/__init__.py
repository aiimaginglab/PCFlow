import torch
import torchvision.transforms as transforms
from datasets_prep.folder_datasets import IR_ImageFolderDataset

from utils.create_degradation import create_degradation


def get_ir_dataset(args):
    
    if args.degradation is not None:
        degradation = create_degradation(args.degradation)
    else:
        degradation = None

    if args.mode == 'train':
        if args.dataset == "ffhq_256":
            train_transform = transforms.Compose(
                [
                    transforms.Resize(args.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            valid_transform = transforms.Compose(
                [
                    transforms.Resize(args.image_size),
                    transforms.ToTensor(),
                ]
            )
            train_dataset = IR_ImageFolderDataset(root_gt=args.train_gt_datadir, root_deg=args.train_deg_datadir, name="ffhq", degradation=degradation, transform=train_transform)
            valid_dataset = IR_ImageFolderDataset(root_gt=args.valid_gt_datadir, root_deg=args.valid_deg_datadir, name="celeba", degradation=degradation, transform=valid_transform)
            return train_dataset, valid_dataset

    elif args.mode == 'test':
        if args.test_dataset == "celeba_test":
            test_transform = transforms.Compose(
                [
                    transforms.Resize(args.image_size),
                    transforms.ToTensor(),
                ]
            )
            dataset = IR_ImageFolderDataset(root_gt=args.test_gt_datadir, root_deg=args.test_deg_datadir, name="celeba", degradation=degradation, transform=test_transform)
            return dataset
