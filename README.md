# Fast Unranking for Preference Revision in Collaborative Filtering Recommendation

## Abstract

In collaborative filtering, it is crucial to efficiently update the model in response to user requests when they express unwanted feedback on previously preferred items. Recommendation unlearning offers a potential solution by removing the influence of outdated interactions without full model retraining. Existing unlearning methods predominantly prioritize privacy protection by erasing specific interaction data. However, this complete erasure paradigm, though vital for privacy compliance, is often unnecessarily stringent for preference revision scenarios, considering that recommendation is a ranking-oriented task. We argue that addressing outdated preferences does not require interaction deletion; rather, demoting target items to lower ranking positions suffices to suppress their unwanted visibility. Motivated by this insight, we introduce $\textit{unranking}$—a practical paradigm that shifts the focus from data erasure to rank demotion without model retraining. To achieve this goal at scale, we propose $\textbf{L2UnRank}$ (Learning to Unrank), a model-agnostic framework that enables fast and effective preference revision of trained recommendation models. Specifically, L2UnRank achieves efficiency by confining influence propagation to local neighborhoods, while ensuring effective unranking through influence-aware parameter updates. Furthermore, when privacy compliance is required, L2UnRank can be transformed to recommendation unlearning by incorporating differential privacy noise during parameter updates. Extensive experiments across five benchmark datasets and three backbones demonstrate that L2UnRank achieves up to 40$\times$ speedup over baselines while maintaining recommendation quality.

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


