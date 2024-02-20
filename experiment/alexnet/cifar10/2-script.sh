
#!/bin/bash

python 2-pruning.py --sensitivity 0.137 --prune './saves/pruned_network-10.ptmodel' > log-2-10.txt
python 2-pruning.py --sensitivity 0.718 --prune './saves/pruned_network-50.ptmodel' > log-2-50.txt
python 2-pruning.py --sensitivity 1.520 --prune './saves/pruned_network-90.ptmodel' > log-2-90.txt
