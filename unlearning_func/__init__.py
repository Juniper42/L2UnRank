# unlearning_func/__init__.py

import time

import torch

from logger import Logger
from trainer import train_model
from utils import get_model

# Import implemented unlearning methods
from .CertifiedRemoval import CertifiedRemoval
from .GSGCF_RU import GSGCF_RU
from .IFRU import IFRU
from .RecEraser import RecEraser
from .SISA import SISA
from .L2UnRank import L2UnRank

logger = Logger.get_logger("Unlearning")


def retrain_unlearning(original_model, data, args, device, user_train_items=None):
    """
    Performs unlearning by retraining the model from scratch on the remaining data.
    This serves as the "gold standard" for unlearning effectiveness but is computationally expensive.

    Args:
        original_model: The original trained model (not used, but kept for consistent API).
        data: The data object, which must contain `train_edge_index_after_remove`.
        args: A dictionary of training arguments.
        device: The computation device.
        user_train_items: Dictionary of user's training items, used for evaluation during training.

    Returns:
        A tuple of (retrained_model, edges_to_remove, unlearning_time).
    """
    logger.info("Starting unlearning via retraining...")

    start_time = time.time()
    edges_to_remove = data.edges_to_remove

    # Create a new model instance with the same architecture
    new_model = get_model(args, data.num_users, data.num_items).to(device)

    # Temporarily replace the training data with the data after removal
    logger.info("Swapping training data and starting retraining...")
    original_train_edge_index = data.train_edge_index
    data.train_edge_index = data.train_edge_index_after_remove

    # Retrain the model on the modified dataset
    retrained_model = train_model(
        new_model, data, args, device, user_train_items=user_train_items
    )

    # Restore the original training data for subsequent evaluation steps
    data.train_edge_index = original_train_edge_index
    unlearning_time = time.time() - start_time
    logger.info(f"Retraining finished. Time taken: {unlearning_time:.2f} seconds.")

    return retrained_model, edges_to_remove, unlearning_time


def gsgcf_ru_unlearning(original_model, data, args, device):
    """
    Performs unlearning using the GSGCF-RU method.
    """
    logger.info("Starting unlearning via GSGCF-RU...")
    gsgcf_ru = GSGCF_RU(original_model, data, args, device)
    return gsgcf_ru.unlearn()


def receraser_unlearning(original_model, data, args, device):
    """
    Performs unlearning using the RecEraser method.
    RecEraser is an efficient framework that uses data partitioning and adaptive aggregation.
    """
    logger.info("Starting unlearning via RecEraser...")
    receraser = RecEraser(original_model, data, args, device)
    return receraser.unlearn()


def utur_unlearning(original_model, data, args, device):
    """
    Performs unlearning using the L2UnRank method.
    L2UnRank is a universal unlearning framework for recommendation, extending GURec to non-GNN models.
    """
    logger.info("Starting unlearning via L2UnRank...")
    utur = L2UnRank(original_model, data, args, device)
    return utur.unlearn()


def ifru_unlearning(original_model, data, args, device):
    """
    Performs unlearning using the IFRU method.
    IFRU is an influence function-based framework that calculates direct and overflow influence.
    """
    logger.info("Starting unlearning via IFRU...")
    ifru = IFRU(original_model, data, args, device)
    return ifru.unlearn()


def sisa_unlearning(original_model, data, args, device):
    """
    Performs unlearning using the SISA (Sharded, Isolated, Sliced, and Aggregated) method.
    This method shards data into multiple independent sub-models and only retrains the
    shards containing the data to be forgotten.
    """
    logger.info("Starting unlearning via SISA...")
    sisa = SISA(original_model, data, args, device)
    return sisa.unlearn()


def certified_removal_unlearning(original_model, data, args, device):
    """
    Performs unlearning using the Certified Removal method.
    This method provides differential privacy guarantees by training a removal-friendly model
    and then using Newton updates for efficient data removal.
    """
    logger.info("Starting unlearning via Certified Removal...")
    certified_removal = CertifiedRemoval(original_model, data, args, device)
    return certified_removal.unlearn()
