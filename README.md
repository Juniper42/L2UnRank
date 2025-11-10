# Learning to Fast Unrank in Collaborative Filtering Recommendation

## Abstract

* **(Please fill in the abstract of your paper here.)**

## Code Structure

```
Project/
├── data/                    # Datasets (ml-1m, yelp2018, amazon-book)
├── models/                  # Recommendation model implementations
├── unlearning_func/         # Unlearning algorithm implementations
├── attack.py                # Membership Inference Attack logic
├── data_loader.py           # Data loading and preprocessing
├── evaluate.py              # Evaluation metrics and functions
├── main.py                  # Main script to run experiments
├── parameters.py            # Command-line argument definitions
├── trainer.py               # Model training logic
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup and Installation

2.  **Create a conda environment:**
    ```bash
    conda create -n l2unrank python=3.10
    ```

3.  **Install dependencies:**
    ```bash
    conda acitvate l2unrank
    pip install -r requirements.txt
    ```

## How to Run

The main script for running all experiments is `main.py`. You can configure the experiment using command-line arguments. Parameters are defined in the `parameters.py`.

运行方法如下所示：

### Retrain

```bash
python main.py \
--method retrain \ 
--dataset ml-1m \
--backbone lightgcn \ 
--lr 0.0001 \
--epoch 1000 \
--batch_size 1024 \
--emb_dim 64 \
--num_layers 2 \
--weight_decay 0.0001 \ 
--neg_samples 1 \
--unlearning_task item \ 
--unlearning_ratio 0.05 \
--test_size 0.2 \
--num_runs 10 \
--output_result_path output_results/ml-1m_lightgcn_0.05n.csv
```

### L2UnRank

```bash
python main.py \
--method l2unrank \
--dataset ml-1m \
--backbone lightgcn \ 
--lr 0.0001 \
--epoch 1000 \
--batch_size 1024 \
--emb_dim 64 \
--num_layers 2 \
--weight_decay 0.0001 \
--neg_samples 1 \
--unlearning_task item \
--unlearning_ratio 0.05 \
--test_size 0.2 \
--influence_hops 1 \ 
--iteration 5 \ 
--scale 0.1 \
--degree_weight 0.5 \
--score_weight 0.5 \
--num_runs 10 \
--output_result_path output_results/ml-1m_lightgcn_0.05n.csv
```
