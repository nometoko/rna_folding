#! /bin/zsh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate rna_folding

input_dir='./input'

datasets=("shujun717/ribonanzanet2d-final" "shujun717/ribonanzanet-weights")

if ! command -v kaggle &> /dev/null; then
    echo "Kaggle is not installed. Please install it or activate the virtual environment."
    exit 1
fi

for dataset in "${datasets[@]}"; do
    echo "Downloading dataset: $dataset"
    # Check if the dataset is already downloaded
    if [ ! -d "${input_dir}/${dataset:t}" ]; then
        # Download the dataset
        mkdir -p "${input_dir}/${dataset:t}"
        kaggle datasets download ${dataset} -p ${input_dir}
        # Unzip the dataset

        unzip "${input_dir}/${dataset:t}.zip" -d "${input_dir}/${dataset:t}"
        # Remove the zip file
        rm "${input_dir}/${dataset:t}.zip"

    else
        echo "Dataset $dataset already exists. Skipping download."
    fi
    echo ""
done
