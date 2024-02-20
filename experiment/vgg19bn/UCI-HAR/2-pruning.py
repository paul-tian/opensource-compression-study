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
from utils.code_util import arguements
from utils.model_util_har import prune_model_eval
from utils.model_util_har import prune_model_train
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

    model = torch.load(args.full)
    model.to(device)

    loss_func = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.tlr, momentum=0.9, weight_decay=5e-4)

    data_path = expanduser("~") + '/DATASET/UCI-HAR/'
    train_loader, test_loader = load(path=data_path, batch_size=args.batch_size)

    print("Eval full model")
    prune_model_eval(model, loss_func, test_loader, device)

    print("--- Pruning ---")
    model.prune_by_threshold(args.sensitivity)
    print("\n")

    print("--- After pruning ---")
    util.print_nonzeros(model)
    print("Eval pruned model")
    prune_model_eval(model, loss_func, test_loader, device)
    print("\n")

    print("--- Retraining ---")
    prune_model_train(num_epochs=args.epochs, model=model, optimizer=optimizer, loss_func=loss_func,
                      train_loader=train_loader, test_loader=test_loader,
                      device=device, adaptiveLR=True)
    print("\n")

    print("--- After Retraining ---")
    util.print_nonzeros(model)
    print("Eval retrained pruned model")
    prune_model_eval(model, loss_func, test_loader, device)
    print("\n")

    torch.save(model, args.prune)
