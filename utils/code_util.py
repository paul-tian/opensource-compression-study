import argparse

def arguements():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for spatial domain datasets (default: 0.01)')
    parser.add_argument('--tlr', type=float, default=0.001, metavar='TLR',
                        help='learning rate for temporal domain datasets (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu', type=int, default=1, choices=[0, 1], metavar='G',
                        help='Choose which GPU will be used (default: 1, the second GPU)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # for pruning
    parser.add_argument('--sensitivity', type=float, default=1.5,
                        help="sensitivity value that is multiplied to layer's std in order to get threshold value")

    # for weight sharing
    parser.add_argument('--fc-bits', type=int, default=4,
                        help="cluster FC layer non-zero weights to (2 ** fc-bit) groups")
    parser.add_argument('--conv-bits', type=int, default=4,
                        help="cluster Conv layer non-zero weights to (2 ** conv-bit) groups")
    parser.add_argument('--kmp', action='store_true', default=False,
                        help='Use K-Mean++ for clustering centroids initialization')

    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--log', type=str, default='./log.txt',
    #                     help='log file name')
    parser.add_argument('--full', default="./saves/full_network.ptmodel", type=str,
                        help='path to full model output')
    parser.add_argument('--prune', default='./saves/pruned_network.ptmodel', type=str,
                        help='path to pruned model output')
    parser.add_argument('--share', default='./saves/shared_network.ptmodel', type=str,
                        help='path to shared weight model output')
    parser.add_argument('--huffman', default='./saves/huffman_network.ptmodel', type=str,
                        help='path to huffman coded model output')

    return parser.parse_args()
