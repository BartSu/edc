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


# KGC -> Unlearning Target (triplet to qa pair)
conda activate vllm
python datasets/tofu/triplet_to_qa.py \
    --kg output/tofu_forget10_qa_target_alignment/iter0/canon_kg.txt \
    --out datasets/tofu/tofu_forget10_qa_target_alignment_kg_iter0_genqa.jsonl