# An example to run EDC (without refinement) on the tofu forget10 dataset

OIE_LLM=mistralai/Mistral-7B-Instruct-v0.2
SD_LLM=mistralai/Mistral-7B-Instruct-v0.2
SC_LLM=mistralai/Mistral-7B-Instruct-v0.2
SC_EMBEDDER=intfloat/e5-mistral-7b-instruct
DATASET=${1:-tofu_forget10_qa}

# target alignment (rebel > wiki-nr > webnlg)
python run.py \
    --oie_llm $OIE_LLM \
    --oie_few_shot_example_file_path "./few_shot_examples/rebel/oie_few_shot_examples.txt" \
    --sd_llm $SD_LLM \
    --sd_few_shot_example_file_path "./few_shot_examples/rebel/sd_few_shot_examples.txt" \
    --sc_llm $SC_LLM \
    --sc_embedder $SC_EMBEDDER \
    --input_text_file_path "./datasets/tofu/${DATASET}.txt" \
    --target_schema_path "./schemas/rebel_schema.csv" \
    --output_dir "./output/${DATASET}_target_alignment" \
    --logging_verbose 