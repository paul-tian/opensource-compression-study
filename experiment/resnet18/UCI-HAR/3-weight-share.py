import os
from os.path import expanduser
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# add project directory to $PYTHONPATH
file_path = os.path.realpath(__file__)
project_path = "/".join(file_path.split("/")[:-4])
sys.path.insert(0, project_path)

from utils import util
from models.resnet18_har import ResNet18
from utils.code_util import arguements
from utils.model_util_har import prune_model_eval
from compress.quantization import apply_weight_sharing_block
from utils.uci_har_loader import load

if __name__ == '__main__':

    args = arguements()  # read arguments from command line

    # Control Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Select Device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    if use_cuda:
        print("Using CUDA!")
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(args.gpu)  # for choosing GPU when have multiple
    else:
        print('Not using CUDA!!!')
    print("\n")

    # Define the model
    model = ResNet18(in_channels=9, num_classes=6, mask=True)
    model.load_state_dict(torch.load(args.prune))
    model.to(device)

    loss_func = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.tlr, momentum=0.9, weight_decay=5e-4)

    data_path = expanduser("~") + '/DATASET/UCI-HAR/'
    train_loader, test_loader = load(path=data_path, batch_size=args.batch_size)

    print('Param Summary')
    util.print_nonzeros(model)

    print('Accuracy before weight sharing')
    prune_model_eval(model, loss_func, test_loader, device)

    print('Apply weight sharing')
    apply_weight_sharing_block(model=model, fc_bits=args.fc_bits, conv_bits=args.conv_bits, kmp=args.kmp)

    print('Param Summary')
    util.print_nonzeros(model)

    print('Accuracy after weight sharing')
    prune_model_eval(model, loss_func, test_loader, device)

    # Save the shared model
    torch.save(model, args.share)
