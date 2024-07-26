import PIL.Image
import numpy
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import json
import os
from pathlib import Path
from PIL import Image

from transformers import DetrFeatureExtractor
from transformers import DetrConfig, DetrForSegmentation

import numpy as np
from shutil import rmtree
import argparse



# Arguments
parser = argparse.ArgumentParser(description='PyTorch Deeplab v3 Example')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


class CocoPanoptic(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_folder, ann_file, feature_extractor):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])

        self.img_folder = img_folder
        self.ann_folder = Path(ann_folder)
        self.ann_file = ann_file
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
        img_path = Path(self.img_folder) / ann_info['file_name']

        img = Image.open(img_path).convert('RGB')

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        encoding = self.feature_extractor(images=img, annotations=ann_info, masks_path=self.ann_folder, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

    def __len__(self):
        return len(self.coco['images'])



class DetrPanoptic(pl.LightningModule):

    def __init__(self, model, lr, lr_backbone, weight_decay):
        super().__init__()

        self.model = model

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

def set_classes():
    #Se definen las clases
    clases = {"Definition": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "traffic light",
                             "bird", "cat", "bottle", "bowl", "chair", "couch", "bed", "desk", "toilet", "laptop",
                             "book", "clock", "teddy bear", "toothbrush", "hair brush", "street sign", "parking meter",
                             "elephant", "bear", "skateboard", "apple", "pizza"],
              "Ids": [1,2,3,4,5,6,7,10, 16, 17, 44, 51, 62, 63, 65, 69, 70, 73, 84, 85, 88, 90, 91, 12, 14, 22, 23, 41, 53, 59]}
    return clases

def preprocessing(panoptic_json_dir, img_dir, dir_objective, img_objective, mode):
    if mode=="train":
        img_list = os.listdir(img_dir) #convención de idx
        img_list = img_list[:3000]
    else:
        img_list = os.listdir(img_dir) #convención de idx
        img_list = img_list[:500]

    classes = set_classes()

    if len(os.listdir(dir_objective)) < len(img_list):
        for idx in tqdm(range(len(img_list)), "Progreso de procesamiento: "):
            img_name =img_list[idx]
            if not os.path.isdir(os.path.join(dir_objective, img_name)):
                with open(panoptic_json_dir) as f:
                    d = json.load(f)
                    d = d["annotations"]
                    info_anotation = next(item for item in d if img_name in item["file_name"])
                    segment_info = info_anotation["segments_info"]
                img_base = cv2.imread(os.path.join(img_dir,img_list[idx]))
                height, width , _ = np.shape(img_base)
                masks = np.ones((height, width, 2 ))*(-1) #First mask is for instance IDs (item categories) and second mask is for background (staff regions and categories ids)
                for each_data in tqdm(segment_info, "Definiendo máscara: "):
                    if each_data["category_id"] in classes["Ids"]:
                        category_id = each_data["category_id"]
                        instance_id = each_data["id"]
                        iscrowd = each_data["iscrowd"]
                        x_min, y_min, delta_x, delta_y = each_data["bbox"]
                        if not iscrowd:
                            masks[x_min:x_min+delta_x, y_min:y_min+delta_y,0] = np.ones(np.shape(masks[x_min:x_min+delta_x, y_min:y_min+delta_y,0]))*instance_id
                            masks[x_min:x_min+delta_x, y_min:y_min+delta_y,1] = np.ones(np.shape(masks[x_min:x_min+delta_x, y_min:y_min+delta_y,1]))*category_id
                        else:
                            masks[x_min:x_min+delta_x, y_min:y_min+delta_y,1] = np.ones(np.shape(masks[x_min:x_min+delta_x, y_min:y_min+delta_y,1]))*category_id
                final_mask = Image.fromarray((masks).astype(numpy.uint8))
                final_mask.save(os.path.join(dir_objective, img_name))
                img_base = Image.fromarray(img_base)
                img_base.save(os.path.join(img_objective, img_name))
            else:
                pass
    else:
        print("El preprocesamiento ya ha sido completado para este conjunto de datos")

def json_generation_subclass_train(panoptic_json_dir, dir_objective):
    with open(panoptic_json_dir) as f:
        d = json.load(f)
        new_json = {"categories":d["categories"], "images":[], "annotations": []}
        images_train = os.listdir(dir_objective)
        for idx in tqdm(range(len(images_train)), "Agregando información de train: "):
            img_name = images_train[idx][:-4]
            info_image = next(item for item in d["images"] if img_name in item["file_name"])
            info_annotation = next(item for item in d["annotations"] if img_name in item["file_name"])
            new_json["annotations"].append(info_annotation)
            new_json["images"].append(info_image)
        with open('data_panoptic_train.json', 'w') as f:
            json.dump(new_json, f)

def json_generation_subclass_val(panoptic_json_dir, dir_objective):
    with open(panoptic_json_dir) as f:
        d = json.load(f)
        new_json = {"categories":d["categories"], "images":[], "annotations": []}
        images_val = os.listdir(dir_objective)
        for idx in tqdm(range(len(images_val)), "Agregando información de val: "):
            img_name = images_val[idx][:-4]
            info_image = next(item for item in d["images"] if img_name in item["file_name"])
            info_annotation = next(item for item in d["annotations"] if img_name in item["file_name"])
            new_json["annotations"].append(info_annotation)
            new_json["images"].append(info_image)
        with open('data_panoptic_val.json', 'w') as f:
            json.dump(new_json, f)




def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoded_input = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoded_input['pixel_values']
    batch['pixel_mask'] = encoded_input['pixel_mask']
    batch['labels'] = labels
    return batch




if __name__ == '__main__':

    # we reduce the size and max_size to be able to fit the batches in GPU memory
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic", size=500, max_size=600)

    json_file_train = '/media/SSD1/vision/coco/annotations/panoptic_train2017.json'
    json_file_train_new = "/home/afmanrique/lab4/segmentation-grupo-4/data_panoptic_train.json"
    img_dir_train = '/media/SSD1/vision/coco/panoptic_train2017'
    img_dir_train_obj = '/home/afmanrique/lab4/segmentation-grupo-4/images_train'

    json_file_val = '/media/SSD1/vision/coco/annotations/panoptic_val2017.json'
    json_file_val_new = "/home/afmanrique/lab4/segmentation-grupo-4/data_panoptic_val.json"
    img_dir_val = '/media/SSD1/vision/coco/panoptic_val2017'
    img_dir_val_obj = '/home/afmanrique/lab4/segmentation-grupo-4/images_val'

    mask_folder = '/home/afmanrique/lab4/segmentation-grupo-4/masking'
    mask_folder_val = '/home/afmanrique/lab4/segmentation-grupo-4/masking_val'

    if len(os.listdir(mask_folder)) < len(os.listdir(img_dir_train)) + len(os.listdir(img_dir_val)):
        print("Iniciando preprocesamiento para set de entrenamiento")
        #preprocessing(json_file_train, img_dir_train, mask_folder, img_dir_train_obj, "train")
        print("Iniciando preprocesamiento para set de validacion")
        #preprocessing(json_file_val, img_dir_val, mask_folder_val, img_dir_val_obj, "val")
        print("Finalizando preprocesamiento")
        #json_generation_subclass_train(json_file_train, img_dir_train_obj)
        #json_generation_subclass_val(json_file_val,img_dir_val_obj)


    dataset = CocoPanoptic(img_folder=img_dir_train_obj,
                                 ann_folder=mask_folder,
                                 ann_file=json_file_train_new,
                                 feature_extractor=feature_extractor)

    # let's split it up into very tiny training and validation sets using random indices
    np.random.seed(42)
    indices = np.random.randint(low=0, high=len(dataset), size=50)
    train_dataset = torch.utils.data.Subset(dataset, indices[:40])
    val_dataset = torch.utils.data.Subset(dataset, indices[40:])

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    pixel_values, target = train_dataset[2]
    print(pixel_values.shape)
    print(target.keys())

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    state_dict = model.state_dict()
    # Remove class weights
    del state_dict["detr.class_labels_classifier.weight"]
    del state_dict["detr.class_labels_classifier.bias"]
    # define new model with custom class classifier
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50-panoptic", num_labels=250)
    model.load_state_dict(state_dict, strict=False)

    model = DetrPanoptic(model=model, lr=args.lr, lr_backbone=1e-5, weight_decay=args.gamma)

    # pick the first training batch
    batch = next(iter(train_dataloader))
    # forward through the model
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
    trainer = Trainer(gpus=1, max_epochs=25, gradient_clip_val=0.1)
    trainer.fit(model)

