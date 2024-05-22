from torch.utils.data import DataLoader
from dataset import CustomData
from torchvision.transforms import ToTensor,Compose,Normalize,Resize
from tqdm.autonotebook import tqdm
import torch.nn as nn
import torch
import os
from buildResnet18 import BasicBlock,ResNet
import shutil
from torch.utils.tensorboard import SummaryWriter
import argparse
def get_args():
    parser = argparse.ArgumentParser(description='build_and_train_resnet18')
    parser.add_argument('--data_path',type=str,default="G:/My Drive/Data/vehicledataset/data")
    parser.add_argument('--learning_rate','-lr',type=int,default=1e-2)
    parser.add_argument('--checkpoint_path','-ckp',type=str,default=None)
    parser.add_argument('--tensorboard_path',type=str,default='Tensorboard')
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--save_path',type=str,default='saved_models')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--freeze_layers',type=int,default=40)

    args = parser.parse_args()
    return args

def main(args):
    os.makedirs(args.save_path,exist_ok=True)
    if args.tensorboard_path :
        if os.path.isdir(args.tensorboard_path):
            shutil.rmtree(args.tensorboard_path)
        os.makedirs(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size = args.batch_size
    num_classes = 5
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=num_classes).to(device)
    if args.freeze_layers:
        count = 0
        for param in model.parameters():
            param.requires_grad = False
            count = count + 1
            if (count == args.freeze_layers):
                break    
    print(model)

    transform = Compose([
                    Resize((224,224)), 
                    ToTensor(), 
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    epochs = args.epochs
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
            ckp = torch.load(args.checkpoint_path)
            start_epoch = ckp['last_epoch']
            model.load_state_dict(ckp['model'])
            optimizer.load_state_dict(ckp['optimizer'])
    else:
        start_epoch = 0
        
    train_data = CustomData(data_dir=os.path.join(args.data_path,'train'),transform=transform)
    val_data = CustomData(data_dir=os.path.join(args.data_path,'val'),transform=transform)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size= batch_size,
        shuffle=True,
        )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size= batch_size,
    )
    
    min_loss = 999
    for epoch in range(start_epoch,start_epoch + epochs):
        model.train(True)
        progress_bar = tqdm(train_loader, colour="cyan")
        running_loss = total = correct = 0

        for i,(images,labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds,labels)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, start_epoch + epochs, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total += labels.size(0)
            with torch.no_grad():
                running_loss += loss.item()
                _, predictions = torch.max(preds, dim=1)
                correct += (predictions == labels).sum().item()
        writer.add_scalar("Train/Loss", running_loss/total, epoch)
        writer.add_scalar("Train/Accuracy", correct/total, epoch)

        print('Training Loss', running_loss/total)
        print('Training Accuracy: ',correct/total)
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
                "optimizer":optimizer.state_dict()
            }
        if(val_loss<min_loss):
            torch.save(checkpoint,os.path.join(args.save_path,'best.pt'))        
        torch.save(checkpoint,os.path.join(args.save_path,'last.pt')) 
            
if __name__ == '__main__':
       args = get_args()
       main(args)