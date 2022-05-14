import os
import sys
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from models.Conv3D import CNN3D, resnet18, resnet34, resnet50, resnet101, r2plus1d_18
from dataset import CSL_Isolated
from train import train_epoch
from validation import val_epoch
from collections import OrderedDict

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

# Path setting
data_path = "/scratch/jiang.so/data/CSL/images"
label_path = "dictionary.txt"
model_path = "checkpoint/rgb_csl_32"
log_path = "log/CSL_resnet2d+1_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/CSL_resnet2d+1_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 500 #100
epochs = 200
# batch_size = 16
batch_size = 48
learning_rate = 1e-5 #1e-4
log_interval = 80
sample_size = 128
sample_duration = 32
attention = False
drop_p = 0.0
hidden1, hidden2 = 512, 256

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Train with 3DCNN
if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
        num_classes=num_classes, train=True, transform=transform)
    val_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
        num_classes=num_classes, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(train_set)+len(val_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)
    # Create model
    # model = CNN3D(sample_size=sample_size, sample_duration=sample_duration, drop_p=drop_p,
    #             hidden1=hidden1, hidden2=hidden2, num_classes=num_classes).to(device)
    # model = resnet18(pretrained=True, progress=True, sample_size=sample_size, sample_duration=sample_duration,
    #                 attention=attention, num_classes=num_classes).to(device)
    # model = resnet50(pretrained=True, progress=True, sample_size=sample_size, sample_duration=sample_duration,
    #                 attention=attention, num_classes=num_classes).to(device)
    model = r2plus1d_18(pretrained=True, num_classes=num_classes).to(device)
    
    ######################
    checkpoint = torch.load('checkpoint/rgb_csl_32/csl_resnet2d+1_epoch030.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove 'module.'
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict)
    start_epoch = 30
    #######################
    print(model)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001)

    # Start training
    logger.info("Training Started".center(60, '#'))
    for epoch in range(epochs):
        if epoch < start_epoch:
            continue
        print('lr: ', get_lr(optimizer))
        # Train the model
        train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)

        # Validate the model
        val_loss = val_epoch(model, criterion, val_loader, device, epoch, logger, writer)
        scheduler.step(val_loss)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "csl_resnet2d+1_epoch{:03d}.pth".format(epoch+1)))
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
