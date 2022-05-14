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
exp_name = 'csl_500_16f'
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
sample_duration = 16
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

    val_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
        num_classes=num_classes, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(val_set)))

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)
    # Create model
    model = r2plus1d_18(pretrained=True, num_classes=num_classes).to(device)
    
    ######################
    checkpoint = torch.load('checkpoint/rgb_csl/slr_resnet2d+1_epoch196.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove 'module.'
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict)
    #######################
    print(model)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy()

    # Start training
    logger.info("Testing Started".center(60, '#'))
    with torch.no_grad():

        # Validate the model
        val_loss = val_epoch(model, criterion, val_loader, device, 0, logger, writer, 'Test', exp_name)


    logger.info("Testing Finished".center(60, '#'))
