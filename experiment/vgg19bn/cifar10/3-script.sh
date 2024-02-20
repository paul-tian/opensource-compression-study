
#!/bin/bash

python 3-weight-share.py --fc-bits 5 --conv-bits 5 --kmp --prune './saves/full_network.ptmodel' --share './saves/shared_network-full-5-kmp.ptmodel' > log-3-full-5-kmp.txt
python 3-weight-share.py --fc-bits 5 --conv-bits 5 --prune './saves/full_network.ptmodel' --share './saves/shared_network-full-5.ptmodel' > log-3-full-5.txt

python 3-weight-share.py --fc-bits 5 --conv-bits 5 --kmp --prune './saves/pruned_network-10.ptmodel' --share './saves/shared_network-10-5-kmp.ptmodel' > log-3-10-5-kmp.txt
python 3-weight-share.py --fc-bits 5 --conv-bits 5 --prune './saves/pruned_network-10.ptmodel' --share './saves/shared_network-10-5.ptmodel' > log-3-10-5.txt

python 3-weight-share.py --fc-bits 5 --conv-bits 5 --kmp --prune './saves/pruned_network-50.ptmodel' --share './saves/shared_network-50-5-kmp.ptmodel' > log-3-50-5-kmp.txt
python 3-weight-share.py --fc-bits 5 --conv-bits 5 --prune './saves/pruned_network-50.ptmodel' --share './saves/shared_network-50-5.ptmodel' > log-3-50-5.txt

python 3-weight-share.py --fc-bits 5 --conv-bits 5 --kmp --prune './saves/pruned_network-90.ptmodel' --share './saves/shared_network-90-5-kmp.ptmodel' > log-3-90-5-kmp.txt
python 3-weight-share.py --fc-bits 5 --conv-bits 5 --prune './saves/pruned_network-90.ptmodel' --share './saves/shared_network-90-5.ptmodel' > log-3-90-5.txt


python 3-weight-share.py --fc-bits 4 --conv-bits 4 --kmp --prune './saves/full_network.ptmodel' --share './saves/shared_network-full-4-kmp.ptmodel' > log-3-full-4-kmp.txt
python 3-weight-share.py --fc-bits 4 --conv-bits 4 --prune './saves/full_network.ptmodel' --share './saves/shared_network-full-4.ptmodel' > log-3-full-4.txt

python 3-weight-share.py --fc-bits 4 --conv-bits 4 --kmp --prune './saves/pruned_network-10.ptmodel' --share './saves/shared_network-10-4-kmp.ptmodel' > log-3-10-4-kmp.txt
python 3-weight-share.py --fc-bits 4 --conv-bits 4 --prune './saves/pruned_network-10.ptmodel' --share './saves/shared_network-10-4.ptmodel' > log-3-10-4.txt

python 3-weight-share.py --fc-bits 4 --conv-bits 4 --kmp --prune './saves/pruned_network-50.ptmodel' --share './saves/shared_network-50-4-kmp.ptmodel' > log-3-50-4-kmp.txt
python 3-weight-share.py --fc-bits 4 --conv-bits 4 --prune './saves/pruned_network-50.ptmodel' --share './saves/shared_network-50-4.ptmodel' > log-3-50-4.txt

python 3-weight-share.py --fc-bits 4 --conv-bits 4 --kmp --prune './saves/pruned_network-90.ptmodel' --share './saves/shared_network-90-4-kmp.ptmodel' > log-3-90-4-kmp.txt
python 3-weight-share.py --fc-bits 4 --conv-bits 4 --prune './saves/pruned_network-90.ptmodel' --share './saves/shared_network-90-4.ptmodel' > log-3-90-4.txt


python 3-weight-share.py --fc-bits 3 --conv-bits 3 --kmp --prune './saves/full_network.ptmodel' --share './saves/shared_network-full-3-kmp.ptmodel' > log-3-full-3-kmp.txt
python 3-weight-share.py --fc-bits 3 --conv-bits 3 --prune './saves/full_network.ptmodel' --share './saves/shared_network-full-3.ptmodel' > log-3-full-3.txt

python 3-weight-share.py --fc-bits 3 --conv-bits 3 --kmp --prune './saves/pruned_network-10.ptmodel' --share './saves/shared_network-10-3-kmp.ptmodel' > log-3-10-3-kmp.txt
python 3-weight-share.py --fc-bits 3 --conv-bits 3 --prune './saves/pruned_network-10.ptmodel' --share './saves/shared_network-10-3.ptmodel' > log-3-10-3.txt

python 3-weight-share.py --fc-bits 3 --conv-bits 3 --kmp --prune './saves/pruned_network-50.ptmodel' --share './saves/shared_network-50-3-kmp.ptmodel' > log-3-50-3-kmp.txt
python 3-weight-share.py --fc-bits 3 --conv-bits 3 --prune './saves/pruned_network-50.ptmodel' --share './saves/shared_network-50-3.ptmodel' > log-3-50-3.txt

python 3-weight-share.py --fc-bits 3 --conv-bits 3 --kmp --prune './saves/pruned_network-90.ptmodel' --share './saves/shared_network-90-3-kmp.ptmodel' > log-3-90-3-kmp.txt
python 3-weight-share.py --fc-bits 3 --conv-bits 3 --prune './saves/pruned_network-90.ptmodel' --share './saves/shared_network-90-3.ptmodel' > log-3-90-3.txt


python 3-weight-share.py --fc-bits 2 --conv-bits 2 --kmp --prune './saves/full_network.ptmodel' --share './saves/shared_network-full-2-kmp.ptmodel' > log-3-full-2-kmp.txt
python 3-weight-share.py --fc-bits 2 --conv-bits 2 --prune './saves/full_network.ptmodel' --share './saves/shared_network-full-2.ptmodel' > log-3-full-2.txt

python 3-weight-share.py --fc-bits 2 --conv-bits 2 --kmp --prune './saves/pruned_network-10.ptmodel' --share './saves/shared_network-10-2-kmp.ptmodel' > log-3-10-2-kmp.txt
python 3-weight-share.py --fc-bits 2 --conv-bits 2 --prune './saves/pruned_network-10.ptmodel' --share './saves/shared_network-10-2.ptmodel' > log-3-10-2.txt

python 3-weight-share.py --fc-bits 2 --conv-bits 2 --kmp --prune './saves/pruned_network-50.ptmodel' --share './saves/shared_network-50-2-kmp.ptmodel' > log-3-50-2-kmp.txt
python 3-weight-share.py --fc-bits 2 --conv-bits 2 --prune './saves/pruned_network-50.ptmodel' --share './saves/shared_network-50-2.ptmodel' > log-3-50-2.txt

python 3-weight-share.py --fc-bits 2 --conv-bits 2 --kmp --prune './saves/pruned_network-90.ptmodel' --share './saves/shared_network-90-2-kmp.ptmodel' > log-3-90-2-kmp.txt
python 3-weight-share.py --fc-bits 2 --conv-bits 2 --prune './saves/pruned_network-90.ptmodel' --share './saves/shared_network-90-2.ptmodel' > log-3-90-2.txt
