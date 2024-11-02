import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision.transforms import ToTensor,Compose,Normalize,Resize
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
import pretrainedmodels
from vit_pytorch import ViT
from transformers import ViTForImageClassification

import torchvision.models as models

from torch import nn
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split
import importlib

pid = os.getpid()
print(f"pid: {pid}")

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train_model", action="store_true")
parser.add_argument("-p", "--predict_model", action="store_true")
parser.add_argument("-d", "--data", type=str, default="data.pkl")
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-cp", "--checkpoint_path", type=str, default="models/best.pt")
parser.add_argument("-op", "--output_path", type=str, default="output.csv")
parser.add_argument("-od", "--output_dir", type=str, default="data")
parser.add_argument("-sp", "--save_path", type=str, default="models")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.003)
parser.add_argument("-fl", "--freeze_layers", type=int, default=30)
# parser.add_argument("-mdl", "--model_module", type=str, default="resnet18")
parser.add_argument("-mdl", "--model_index", type=int, default=0)
parser.add_argument("-n", "--number_classes", type=int, default=16)
parser.add_argument("-s", "--image_size", type=int, default=64)
args = parser.parse_args()

save_path = args.save_path
train_model = True if args.train_model else False
predict_model = True if args.predict_model else False
print(f"mode: train_model, {train_model}; predict_model, {predict_model}")

image_size = [args.image_size, args.image_size]
num_classes = args.number_classes
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = 64
freeze_layers = args.freeze_layers

df = pd.read_pickle(args.data)
train_dataset = df[df["trainTestNum"] == 0].copy()
train_x, eval_x, train_y, eval_y = train_test_split(train_dataset, train_dataset["failureNum"], test_size=0.25, random_state=42)
test_dataset = df[df["trainTestNum"] == 1].copy()

# train_x, eval_x, train_y, eval_y = train_test_split(df, df["failureNum"], test_size=0.25, random_state=1)

def transform_image(input_image):
    # pil_image = Image.fromarray(input_image * 255 / 2)

    rgb_image = np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.uint8)
    color_map = {
        0: [255, 255, 255],
        1: [0, 0, 0],
        2: [255, 0, 0]
    }

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            rgb_image[i, j] = color_map[input_image[i][j]]
    
    pil_image = Image.fromarray(rgb_image)
    return pil_image

class CustomData(Dataset):
    def __init__(self, df, transform = None) -> None:
        self.df = df.reset_index(drop=True)
        self.labels = []
        self.transform = transform
        for index, row in self.df.iterrows():
            self.labels.append(self.df['failureNum'][index])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_map = self.df["waferMap"][index]
        if self.transform:
            image = self.transform(transform_image(image_map))
            
        label = torch.tensor(self.labels[index])
        return image, label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# model_module = importlib.import_module()
# model = model_module.Model

if args.model_index == 0:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features , num_classes)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
elif args.model_index == 1:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features , num_classes)
elif args.model_index == 2:
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features , num_classes)
elif args.model_index == 3:
    model = pretrainedmodels.__dict__['senet154'](pretrained='imagenet')
    if hasattr(model, 'last_linear'):
        num_features = model.last_linear.in_features
        model.last_linear = nn.Linear(num_features , num_classes)
    else: 
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features , num_classes)
elif args.model_index == 4:
    model = ViT(
        image_size=args.image_size, 
        patch_size=32, 
        num_classes=num_classes, 
        dim=1024, 
        depth=6, 
        heads=8, 
        mlp_dim=2048, 
        dropout=0.1, 
        emb_dropout=0.1 
    )
elif args.model_index == 5:
    model = ViTForImageClassification.from_pretrained(
        "vit-huge-patch14-224-in21k", 
        image_size=args.image_size,
        num_labels=num_classes, 
        local_files_only=True
    )
elif args.model_index == 6:
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    hidden_dim = model.heads.head.in_features
    model.heads.head = nn.Linear(hidden_dim, num_classes)

model = model.to(device)

transform = Compose([
    Resize((image_size[0], image_size[1])), 
    ToTensor(), 
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # Normalize((0.449,), (0.226,))
])

if train_model: 
    # print(model)
    output_dir = os.path.join(args.output_dir, str(pid))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    loss_weight = torch.tensor([1,5,1,1,1,1,5,1,1,5,1,1,1,1,5,1], device=device, dtype=torch.float32)

    with open(os.path.join(output_dir, "param.txt"), "w") as f:
        f.writelines(f"model: {model}\n")
        f.writelines(f" ========================= \n")
        f.writelines(f"model: {args.model_index}\n")
        f.writelines(f"freeze_layers: {freeze_layers}\n")
        f.writelines(f"epochs: {epochs}\n")
        f.writelines(f"num_classes: {num_classes}\n")
        f.writelines(f"loss_weight: {loss_weight}")

    cnt = 0
    for name, param in model.named_parameters():
        cnt += 1
        print(f"{cnt}: Name: {name}, Type: {param.dtype}, Shape: {param.size()}")

    cnt = 0
    for param in model.parameters():
        if cnt == freeze_layers:
            break
        param.requires_grad = False
        cnt += 1

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
    # loss_fn = nn.CrossEntropyLoss()

    # train_data = CustomData(train_dataset, transform=transform)
    train_data = CustomData(train_x, transform=transform)
    train_loader = DataLoader(
        dataset= train_data,
        batch_size= batch_size,
        shuffle= True,
    )

    val_data = CustomData(eval_x, transform=transform)
    val_loader = DataLoader(
        dataset = val_data,
        batch_size = batch_size
    )
    min_loss = 9999
    for epoch in range(epochs):
        model.train(True)
        progress_bar = tqdm(train_loader, colour="cyan")
        running_loss = total = correct = 0

        for i,(images,labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds,labels)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, epochs, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            with torch.no_grad():
                running_loss += loss.item()
                _, predictions = torch.max(preds, dim=1)
                correct += (predictions == labels).sum().item()

        training_loss = running_loss/total
        training_acc = correct/total
        print('Training Loss', training_loss)
        print('Training Accuracy: ', training_acc)

        model.eval()
        progress_bar = tqdm(val_loader, colour="yellow")
        val_loss = correct = total = 0
        for (images,labels) in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = model(images)
                _, predictions = torch.max(preds, dim=1)
                correct += (predictions == labels).sum().item()
            loss = loss_fn(preds,labels)
            val_loss = val_loss + loss.item()
            total += labels.size(0)
        print('Val loss:',val_loss/total)
        print('Val Accuracy: ',correct/total)

        checkpoint  = {
            "model":model.state_dict(),
            "last_epoch": epoch+1,
            "optimizer":optimizer.state_dict(),
            "val_acc":correct/total
        }

        if(val_loss<min_loss):
            min_loss = val_loss
            torch.save(checkpoint,os.path.join(output_dir, f'best_{pid}.pt'))        
        torch.save(checkpoint,os.path.join(output_dir, f'last_{pid}.pt')) 

        # torch.save(checkpoint, os.path.join(save_path, f"epoch_{epoch}.pt"))
elif predict_model:
    # pd.options.copy_on_write = True
    ckp = torch.load(args.checkpoint_path)
    model.load_state_dict(ckp['model'])
    model.eval()

    # test_dataset.loc["eval_result"] = None
    for index, row in eval_x.iterrows():

        image_map = eval_x['waferMap'][index]
        image = transform(transform_image(image_map))
        image = image.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(image.unsqueeze(0))  # Add batch dimension
            print(pred)
        # max_value, max_indice = torch.max(pred, dim=1)
        # if max_value.item() > 0.4:
        #     test_dataset.loc[index, "eval_result"] = int(max_indice.item())
        # else:
        #     test_dataset.loc[index, "eval_result"] = num_classes
        prediction = int(torch.max(pred, dim=1).indices)
        eval_x.loc[index, "eval_result"] = prediction
    eval_x.to_csv(args.output_path)
else:
    res = pd.read_csv(args.output_path)

    res["eval_result1"] = res["eval_result"].apply(lambda x: x-8 if x > 7 else x)
    cnt = {i : 0 for i in range(num_classes)}
    sum = {i : 0 for i in range(num_classes)}
    cls = {}
    for index, row in res.iterrows():
        # if row["failureNum"] not in cls:
        #     cls[row["failureNum"]] = row["failureType"]
        # sum[row["failureNum"]] += 1        
        # if row["failureNum"] == row["eval_result"]:
        #     cnt[row["failureNum"]] += 1
        sum[row["failureNum"]] += 1        
        if row["failureNum"] == row["eval_result"]:
            cnt[row["failureNum"]] += 1
    for k in range(16):
        if sum[k] == 0:
            continue
        print(f"{k}: {cnt[k] / sum[k]}")
    
    print("===========")

    cnt = {i : 0 for i in range(num_classes)}
    sum = {i : 0 for i in range(num_classes)}
    cls = {}
    for index, row in res.iterrows():
        # if row["failureNum"] not in cls:
        #     cls[row["failureNum"]] = row["failureType"]
        # sum[row["failureNum"]] += 1        
        # if row["failureNum"] == row["eval_result"]:
        #     cnt[row["failureNum"]] += 1
        sum[row["oldFailureNum"]] += 1        
        if row["oldFailureNum"] == row["eval_result1"]:
            cnt[row["oldFailureNum"]] += 1
    for k in range(8):
        if sum[k] == 0:
            continue
        print(f"{k}: {cnt[k] / sum[k]}")


