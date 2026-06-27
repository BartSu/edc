# download datasets
cd /media/volume/asr/edc/datasets/tofu
python tofu.py --config forget10 --mode qa
python tofu.py --config full --mode qa
python tofu.py --config real_authors --mode qa
python tofu.py --config world_facts --mode qa

# KGC
cd /media/volume/asr/edc
conda activate edc
## target alignment (rebel)
bash run-tofu-forget10.sh

## self canonicalization