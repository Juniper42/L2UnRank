import argparse


def str2bool(v):
    """Converts a string to a boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Config:
    """Configuration class for managing command-line arguments."""

    @staticmethod
    def get_config():
        """Gets the command-line argument configuration."""
        parser = argparse.ArgumentParser(
            description="Comparative Experiments of Unlearning Methods in Recommender Systems"
        )

        # Core Parameters
        parser.add_argument(
            "--method",
            type=str,
            default="retrain",
            choices=[
                "retrain",
                "gsgcf-ru",
                "receraser",
                "l2unrank",
                "ifru",
                "sisa",
                "certified_removal",
            ],
            help="Unlearning method to use.",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="ml-1m",
            choices=["ml-1m", "yelp2018", "amazon-book"],
            help="Dataset to use.",
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="lightgcn",
            choices=[
                "lightgcn",
                "gat",
                "graphsage",
                "mlp",
                "gmf",
                "neumf",
                "wmf",
                "gcn",
                "gin",
            ],
            help="Backbone model for recommendations.",
        )
        parser.add_argument(
            "--is_attack",
            type=str2bool,
            default=True,
            help="Whether to perform a Membership Inference Attack.",
        )
        parser.add_argument(
            "--evaluate_forgetting",
            type=str2bool,
            default=True,
            help="Whether to evaluate the forgetting effect.",
        )
        parser.add_argument(
            "--split_method",
            type=str,
            default="temporal",
            help="Dataset splitting method (e.g., temporal).",
        )
        parser.add_argument(
            "--k", type=int, default=10, help="Size of the recommendation list (top-k)."
        )
        parser.add_argument(
            "--use_cache",
            type=str2bool,
            default=True,
            help="Whether to use cached data.",
        )
        parser.add_argument(
            "--use_model_cache",
            type=str2bool,
            default=True,
            help="Whether to use cached model weights.",
        )
        parser.add_argument(
            "--output_result_path",
            type=str,
            default="",
            help="Path to save the experiment results in a CSV file.",
        )
        parser.add_argument(
            "--num_runs",
            type=int,
            default=10,
            help="Number of experimental runs for averaging results.",
        )

        # Training Parameters
        parser.add_argument(
            "--lr", type=float, default=1e-3, help="Learning rate for the main model."
        )
        parser.add_argument(
            "--epoch", type=int, default=100, help="Number of training epochs."
        )
        parser.add_argument(
            "--batch_size", type=int, default=4096, help="Training batch size."
        )
        parser.add_argument(
            "--emb_dim", type=int, default=256, help="Embedding dimension."
        )
        parser.add_argument(
            "--num_layers", type=int, default=3, help="Number of GNN layers."
        )
        parser.add_argument(
            "--dropout_rate", type=float, default=0.2, help="Dropout rate."
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-4,
            help="Weight decay (L2 regularization).",
        )
        parser.add_argument(
            "--test_size",
            type=float,
            default=0.2,
            help="Proportion of the dataset to use for testing.",
        )
        parser.add_argument(
            "--val_size",
            type=float,
            default=0.0,
            help="Proportion for validation set. If 0, test set is used for validation.",
        )
        parser.add_argument(
            "--eval_interval",
            type=int,
            default=5,
            help="Interval (in epochs) for evaluation.",
        )
        parser.add_argument(
            "--early_stop",
            type=str2bool,
            default=True,
            help="Whether to enable early stopping.",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=5,
            help="Patience for early stopping.",
        )
        parser.add_argument(
            "--early_stop_metric",
            type=str,
            default="recall@k",
            choices=["recall@k", "ndcg@k"],
            help="Metric for early stopping.",
        )
        parser.add_argument(
            "--scheduler_type",
            type=str,
            default="cosine",
            choices=["cosine", "plateau", "exponential", "default"],
            help="Learning rate scheduler type.",
        )
        parser.add_argument(
            "--lr_gamma",
            type=float,
            default=0.95,
            help="Decay factor for the exponential learning rate scheduler.",
        )
        parser.add_argument(
            "--neg_samples",
            type=int,
            default=20,
            help="Number of negative samples per positive sample.",
        )
        parser.add_argument(
            "--reg_weight",
            type=float,
            default=1e-3,
            help="Weight for L2 regularization loss.",
        )
        parser.add_argument(
            "--margin",
            type=float,
            default=0.0,
            help="Margin for BPR loss.",
        )
        parser.add_argument(
            "--loss_function",
            type=str,
            default="bpr",
            choices=["bpr", "pointwise_bce", "infonce"],
            help="Loss function type.",
        )
        parser.add_argument(
            "--loss_temperature",
            type=float,
            default=0.2,
            help="Temperature parameter for InfoNCE loss.",
        )

        # MIA Parameters
        parser.add_argument(
            "--attack_batch_size",
            type=int,
            default=512,
            help="Batch size for the attack model.",
        )
        parser.add_argument(
            "--attack_epochs",
            type=int,
            default=100,
            help="Number of training epochs for the attack model.",
        )
        parser.add_argument(
            "--attack_lr",
            type=float,
            default=0.001,
            help="Learning rate for the attack model.",
        )

        # Unlearning Parameters
        parser.add_argument(
            "--unlearning_ratio",
            type=float,
            default=0.1,
            help="Proportion of users or items to be unlearned.",
        )
        parser.add_argument(
            "--unlearning_task",
            type=str,
            default="interaction",
            choices=["interaction", "item"],
            help="Type of unlearning task: 'interaction' or 'item'.",
        )
        parser.add_argument(
            "--unlearning_interaction_ratio",
            type=float,
            default=0.2,
            help="Proportion of a user's interactions to remove in 'interaction' unlearning.",
        )

        # General parameters that might be used by multiple methods
        parser.add_argument(
            "--influence_hops",
            type=int,
            default=1,
            help="Number of hops for influence calculation.",
        )
        parser.add_argument(
            "--iteration",
            type=int,
            default=10,
            help="Number of iterations for influence approximation.",
        )
        parser.add_argument(
            "--damp",
            type=float,
            default=0.0001,
            help="Damping factor for influence calculation.",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=0.1,
            help="Scaling factor for influence calculation.",
        )
        parser.add_argument(
            "--use_high_impact",
            type=str2bool,
            default=True,
            help="Whether to use high-impact edges for unlearning (e.g., in GIF-like methods).",
        )
        parser.add_argument(
            "--degree_weight",
            type=float,
            default=0.5,
            help="Weight for node degree in impact scoring.",
        )
        parser.add_argument(
            "--score_weight",
            type=float,
            default=0.5,
            help="Weight for interaction score in impact scoring.",
        )
        parser.add_argument(
            "--high_impact_ratio",
            type=float,
            default=1.0,
            help="Ratio of high-impact edges to consider.",
        )

        # GSGCF-RU Parameters
        parser.add_argument(
            "--unlearning_epochs",
            type=int,
            default=100,
            help="Number of epochs for GSGCF-RU fine-tuning.",
        )
        parser.add_argument(
            "--unlearning_lr",
            type=float,
            default=1e-3,
            help="Learning rate for GSGCF-RU fine-tuning.",
        )
        parser.add_argument(
            "--lambda",
            type=float,
            default=0.5,
            help="Weight to balance consistency and causality in GSGCF-RU.",
        )
        parser.add_argument(
            "--unlearning_neg_samples",
            type=int,
            default=5,
            help="Number of negative samples for GSGCF-RU.",
        )

        # RecEraser Parameters
        parser.add_argument(
            "--attention_size",
            type=int,
            default=None,
            help="Attention mechanism dimension for RecEraser.",
        )
        parser.add_argument(
            "--num_partitions",
            type=int,
            default=3,
            help="Number of partitions for RecEraser.",
        )
        parser.add_argument(
            "--partition_type",
            type=str,
            default="interaction",
            choices=["user", "item", "interaction"],
            help="Partitioning strategy for RecEraser.",
        )
        parser.add_argument(
            "--sub_epochs",
            type=int,
            default=10,
            help="Number of training epochs for sub-models in RecEraser.",
        )
        parser.add_argument(
            "--agg_epochs",
            type=int,
            default=5,
            help="Number of aggregation epochs in RecEraser.",
        )
        parser.add_argument(
            "--sub_lr",
            type=float,
            default=0.001,
            help="Learning rate for sub-models in RecEraser.",
        )
        parser.add_argument(
            "--att_lr",
            type=float,
            default=0.001,
            help="Learning rate for the attention mechanism in RecEraser.",
        )

        # IFRU Parameters
        parser.add_argument(
            "--pruning_rates",
            nargs="+",
            type=float,
            default=[1.0, 0.5, 0.25],
            help="Pruning rates for IFRU, e.g., --pruning_rates 0.8 0.6 0.4",
        )

        # SISA Parameters
        parser.add_argument(
            "--sisa_num_shards",
            type=int,
            default=5,
            help="Number of data shards for SISA.",
        )
        parser.add_argument(
            "--sisa_num_slices",
            type=int,
            default=1,
            help="Number of data slices for SISA, supporting incremental training.",
        )
        parser.add_argument(
            "--sisa_aggregation",
            type=str,
            default="uniform",
            choices=["uniform", "weighted"],
            help="Aggregation strategy for SISA: 'uniform' or 'weighted'.",
        )

        # CertifiedRemoval Parameters
        parser.add_argument(
            "--certified_lam",
            type=float,
            default=1e-4,
            help="L2 regularization parameter for CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_std",
            type=float,
            default=10.0,
            help="Target perturbation standard deviation for CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_num_steps",
            type=int,
            default=100,
            help="Number of LBFGS optimization steps for CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_subsample_ratio",
            type=float,
            default=1.0,
            help="Negative sampling ratio for CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_batch_size_hessian",
            type=int,
            default=50000,
            help="Batch size for Hessian matrix computation in CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_finetune_lr",
            type=float,
            default=1e-4,
            help="Learning rate for the fine-tuning step in CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_finetune_epochs",
            type=int,
            default=5,
            help="Number of epochs for the fine-tuning step in CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_train_mode",
            type=str,
            default="ovr",
            choices=["ovr", "binary"],
            help="Training mode for CertifiedRemoval: 'ovr' (One-vs-Rest) or 'binary'.",
        )

        return parser
