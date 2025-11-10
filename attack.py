import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from logger import Logger
from models import MIAttackModel

logger = Logger.get_logger("MIAttack")


class MembershipInferenceAttack:
    """
    Implements a black-box Membership Inference Attack (MIA) to evaluate privacy leakage.
    """

    def __init__(self, original_model, unlearning_model, data, device, args):
        self.original_model = original_model
        self.unlearning_model = unlearning_model
        self.data = data
        self.device = device
        self.args = args

        # Attack model training parameters
        self.attack_batch_size = args.get("attack_batch_size", 256)
        self.attack_epochs = args.get("attack_epochs", 50)
        self.attack_lr = args.get("attack_lr", 0.001)

    def get_model_predictions(self, model, users, items, edge_index=None):
        """
        Retrieves prediction scores from a recommendation model in a black-box manner.

        Args:
            model: The recommendation model.
            users: Tensor of user indices.
            items: Tensor of item indices (raw, with user count offset).
            edge_index: Edge index required by some GNN models.

        Returns:
            A tensor of prediction scores.
        """
        predictions = []
        batch_size = 1024  # Batching to prevent OOM errors

        model.eval()
        with torch.no_grad():
            for i in range(0, users.size(0), batch_size):
                batch_users = users[i : i + batch_size]
                batch_items = items[i : i + batch_size]
                # Item indices need to be adjusted to be in the range [0, num_items-1]
                adjusted_items = batch_items - self.data.num_users

                try:
                    # Primary method: use the model's forward pass
                    batch_preds = model.forward(batch_users, adjusted_items)
                    predictions.append(batch_preds.cpu())
                except Exception as e:
                    logger.warning(
                        f"Prediction via forward() failed: {e}. Trying fallback."
                    )
                    # Fallback method: directly compute dot product of embeddings
                    try:
                        user_emb, item_emb = model.get_embeddings()
                        batch_user_emb = user_emb[batch_users]
                        batch_item_emb = item_emb[adjusted_items]
                        batch_preds = torch.sum(batch_user_emb * batch_item_emb, dim=1)
                        predictions.append(batch_preds.cpu())
                    except Exception as e2:
                        logger.error(
                            f"Fallback prediction method also failed: {e2}. Returning zero scores."
                        )
                        predictions.append(torch.zeros(batch_users.size(0)))

        return torch.cat(predictions, dim=0)

    def extract_features_blackbox(self, target_edges, is_member):
        """
        Extracts features for the attack model based on prediction scores (black-box).

        Args:
            target_edges: A list of (user, item) tuples.
            is_member: Boolean indicating if these edges are members of the training set.

        Returns:
            A tuple of (features, labels) tensors.
        """
        if not target_edges:
            return None, None

        users = torch.tensor([u for u, i in target_edges], device=self.device)
        items = torch.tensor([i for u, i in target_edges], device=self.device)

        # Get prediction scores from both original and unlearned models
        orig_scores = self.get_model_predictions(
            self.original_model, users, items, edge_index=self.data.train_edge_index
        ).to(self.device)

        unlearning_edge_index = getattr(
            self.data, "train_edge_index_after_remove", self.data.train_edge_index
        )
        unlearn_scores = self.get_model_predictions(
            self.unlearning_model, users, items, edge_index=unlearning_edge_index
        ).to(self.device)

        # Construct features based on the prediction scores
        features = self.compute_score_based_features(orig_scores, unlearn_scores)
        labels = (
            torch.ones(len(target_edges))
            if is_member
            else torch.zeros(len(target_edges))
        )

        return features.cpu(), labels

    def compute_score_based_features(self, orig_scores, unlearn_scores):
        """
        Computes a feature vector from the original and unlearned model scores.

        Args:
            orig_scores: Prediction scores from the original model.
            unlearn_scores: Prediction scores from the unlearned model.

        Returns:
            A tensor of computed features.
        """
        # Basic score features
        score_diff = orig_scores - unlearn_scores
        score_abs_diff = torch.abs(score_diff)
        score_ratio = orig_scores / (torch.abs(unlearn_scores) + 1e-8)

        # Statistical features
        score_sum = orig_scores + unlearn_scores
        score_product = orig_scores * unlearn_scores

        # Distributional features based on sigmoid-normalized scores
        orig_scores_normalized = torch.sigmoid(orig_scores)
        unlearn_scores_normalized = torch.sigmoid(unlearn_scores)
        prob_diff = orig_scores_normalized - unlearn_scores_normalized

        # Relative change feature
        relative_change = score_diff / (torch.abs(orig_scores) + 1e-8)

        # Combine all features
        feature_list = [
            orig_scores,
            unlearn_scores,
            score_diff,
            score_abs_diff,
            score_ratio,
            score_sum,
            score_product,
            orig_scores_normalized,
            unlearn_scores_normalized,
            prob_diff,
            relative_change,
        ]

        return torch.stack(feature_list, dim=1)

    def prepare_attack_dataset(self, edges_to_remove=None):
        """
        Prepares the dataset for the MIA attack.
        This involves identifying member (forgotten) and non-member edges and extracting features.

        Args:
            edges_to_remove: The edges that were removed during unlearning.

        Returns:
            A tuple of (train_features, train_labels, test_features, test_labels).
        """
        logger.info("Preparing black-box MIA dataset...")

        # Identify the set of edges that were removed (these are the 'member' samples)
        if edges_to_remove is None or edges_to_remove.numel() == 0:
            logger.error("No edges_to_remove provided for MIA. Aborting.")
            return None, None, None, None

        removed_edges = set(tuple(e) for e in edges_to_remove.t().cpu().tolist())
        logger.info(
            f"Identified {len(removed_edges)} unique edges to be forgotten (members)."
        )

        # Cap the number of member edges to prevent excessive computation
        max_member_samples = 10000
        if len(removed_edges) > max_member_samples:
            removed_edges = set(random.sample(list(removed_edges), max_member_samples))
            logger.info(f"Capped member samples to {len(removed_edges)}.")

        # Identify all edges present in the original training set
        original_train_edges = set(
            tuple(e) for e in self.data.train_edge_index.t().cpu().tolist()
        )

        # Generate non-member edges: edges not in the original training set
        num_non_members = len(removed_edges)
        non_member_edges = set()
        max_attempts = num_non_members * 5
        logger.info(f"Generating {num_non_members} non-member edges...")

        while len(non_member_edges) < num_non_members and max_attempts > 0:
            random_users = torch.randint(0, self.data.num_users, (num_non_members,))
            random_items = torch.randint(
                self.data.num_users,
                self.data.num_users + self.data.num_items,
                (num_non_members,),
            )

            for i in range(num_non_members):
                u, v = random_users[i].item(), random_items[i].item()
                edge = (u, v)
                if edge not in original_train_edges and edge not in non_member_edges:
                    non_member_edges.add(edge)
                if len(non_member_edges) >= num_non_members:
                    break
            max_attempts -= num_non_members

        logger.info(f"Generated {len(non_member_edges)} non-member edges.")

        # Extract features for both sets
        logger.info("Extracting features... This may take a while.")
        member_features, member_labels = self.extract_features_blackbox(
            list(removed_edges), is_member=True
        )
        non_member_features, non_member_labels = self.extract_features_blackbox(
            list(non_member_edges), is_member=False
        )

        if member_features is None or non_member_features is None:
            logger.error("Feature extraction failed. Cannot proceed with the attack.")
            return None, None, None, None

        # Combine, normalize, and split the dataset
        all_features = torch.cat([member_features, non_member_features], dim=0)
        all_labels = torch.cat([member_labels, non_member_labels], dim=0)

        logger.info(
            f"Feature extraction complete. Total samples: {all_features.shape[0]}, Feature dim: {all_features.shape[1]}"
        )

        # Normalize features
        feature_mean = all_features.mean(dim=0, keepdim=True)
        feature_std = all_features.std(dim=0, keepdim=True) + 1e-6
        all_features = (all_features - feature_mean) / feature_std

        # Shuffle and split into training and testing sets
        indices = torch.randperm(all_features.size(0))
        train_size = int(0.8 * all_features.size(0))
        train_indices, test_indices = indices[:train_size], indices[train_size:]

        train_features, test_features = (
            all_features[train_indices],
            all_features[test_indices],
        )
        train_labels, test_labels = all_labels[train_indices], all_labels[test_indices]

        logger.info(
            f"Dataset prepared: {len(train_labels)} training samples, {len(test_labels)} test samples."
        )
        return train_features, train_labels, test_features, test_labels

    def train_attack_model(self, train_features, train_labels):
        """
        Trains the MIA attack model.
        """
        logger.info(f"Training MIA model on {self.device}...")

        train_features = train_features.to(self.device)
        train_labels = train_labels.to(self.device)

        attack_model = MIAttackModel(train_features.size(1)).to(self.device)
        optimizer = torch.optim.Adam(
            attack_model.parameters(), lr=self.attack_lr, weight_decay=1e-5
        )

        # Handle data imbalance by assigning more weight to the minority class in the loss
        pos_count = (train_labels == 1).sum().item()
        neg_count = (train_labels == 0).sum().item()
        pos_weight = (
            torch.tensor([neg_count / pos_count], device=self.device)
            if pos_count > 0
            else None
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        attack_model.train()
        for epoch in range(self.attack_epochs):
            total_loss = 0
            num_batches = 0

            # Shuffle data each epoch
            perm = torch.randperm(train_features.size(0))
            for i in range(0, train_features.size(0), self.attack_batch_size):
                indices = perm[i : i + self.attack_batch_size]
                batch_features, batch_labels = (
                    train_features[indices],
                    train_labels[indices],
                )

                optimizer.zero_grad()
                logits = attack_model(batch_features).squeeze()
                loss = criterion(logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(attack_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.attack_epochs}, Avg Loss: {total_loss / num_batches:.4f}"
                )

        logger.info("MIA model training complete.")
        return attack_model.to(self.device)

    def evaluate_attack(self, attack_model, test_features, test_labels):
        """
        Evaluates the performance of the trained MIA attack model.
        """
        logger.info("Evaluating MIA model...")
        attack_model.eval()
        test_features = test_features.to(self.device)
        test_labels = test_labels.cpu().numpy()

        with torch.no_grad():
            # Use sigmoid since the model outputs logits and we used BCEWithLogitsLoss
            outputs = torch.sigmoid(attack_model(test_features).squeeze()).cpu().numpy()

        predictions = (outputs > 0.5).astype(int)

        # Calculate standard classification metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        auc = roc_auc_score(test_labels, outputs)
        ap = average_precision_score(test_labels, outputs)

        # Calculate additional privacy-specific metrics
        member_indices = test_labels == 1
        non_member_indices = test_labels == 0

        member_confidence = (
            outputs[member_indices].mean() if member_indices.any() else 0
        )
        non_member_confidence = (
            outputs[non_member_indices].mean() if non_member_indices.any() else 0
        )
        prob_gap = abs(member_confidence - non_member_confidence)

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "average_precision": ap,
            "member_confidence": member_confidence,
            "non_member_confidence": non_member_confidence,
            "probability_gap": prob_gap,
        }
        return results

    def run_attack(self, edges_to_remove=None):
        """
        Executes the full black-box Membership Inference Attack pipeline.
        """
        logger.info("Starting black-box Membership Inference Attack...")

        # Set random seeds for reproducibility
        seed = self.args.get("random_seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 1. Prepare dataset
        dataset = self.prepare_attack_dataset(edges_to_remove)
        if dataset is None:
            return None
        train_features, train_labels, test_features, test_labels = dataset

        # 2. Train attack model
        attack_model = self.train_attack_model(train_features, train_labels)

        # 3. Evaluate attack model
        attack_results = self.evaluate_attack(attack_model, test_features, test_labels)

        # 4. Log results
        logger.info("\n--- Black-Box MIA Results ---")
        logger.info(f"  Accuracy:           {attack_results['accuracy']:.4f}")
        logger.info(f"  Precision:          {attack_results['precision']:.4f}")
        logger.info(f"  Recall:             {attack_results['recall']:.4f}")
        logger.info(f"  F1-Score:           {attack_results['f1_score']:.4f}")
        logger.info(f"  AUC:                {attack_results['auc']:.4f}")
        logger.info(f"  Average Precision:  {attack_results['average_precision']:.4f}")
        logger.info(f"  Confidence Gap:     {attack_results['probability_gap']:.4f}")
        logger.info("---------------------------\n")

        return attack_results
