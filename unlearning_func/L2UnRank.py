import copy
import time

import numpy as np
import torch
import torch.nn.utils
from torch.autograd import grad

from logger import Logger
from trainer import train_model
from utils import determine_is_gnn

logger = Logger.get_logger("Unlearning(L2UnRank)")


class L2UnRank:
    def __init__(self, model, data, args, device):
        self.model = model
        self.data = data
        self.args = args
        self.device = device
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def _get_current_embeddings(self):
        """Helper to get embeddings, handling both GNN and non-GNN models."""
        if determine_is_gnn(self.args.get("backbone")):
            return self.model(edge_index=self.data.train_edge_index)
        return self.model.get_embeddings()

    def _compute_loss_grad(self, model, edge_index):
        """Computes the BPR loss gradient for a given set of edges."""
        users = edge_index[0]
        pos_items = edge_index[1] - self.data.num_users
        neg_items = torch.randint(
            0, self.data.num_items, (users.size(0),), device=self.device
        )

        user_emb, item_emb = model.get_embeddings()

        u_emb = user_emb[users]
        pos_i_emb = item_emb[pos_items]
        neg_i_emb = item_emb[neg_items]

        pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)

        loss = -torch.logsigmoid(pos_scores - neg_scores).mean()

        grads = grad(loss, self.params, create_graph=True, allow_unused=True)
        return [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.params)
        ]

    def _hessian_vector_product(self, loss_grads, v):
        """Computes the Hessian-vector product (HVP)."""
        dot_product = torch.sum(
            torch.stack([torch.sum(g * vi) for g, vi in zip(loss_grads, v)])
        )
        hvp = grad(dot_product, self.params, retain_graph=True, allow_unused=True)
        return [
            h if h is not None else torch.zeros_like(p)
            for h, p in zip(hvp, self.params)
        ]

    def _conjugate_gradient_solve(self, b_grads):
        """Approximates H^-1 * b using the Conjugate Gradient algorithm."""
        # Using the gradient w.r.t. the remaining data for HVP calculation
        loss_grads_D_c = self._compute_loss_grad(
            self.model, self.data.train_edge_index_after_remove
        )

        x = [torch.zeros_like(p) for p in self.params]
        r = [b_g.clone() for b_g in b_grads]
        p = [r_g.clone() for r_g in b_grads]
        rs_old = sum(torch.sum(r_g * r_g) for r_g in r)

        for _ in range(self.args.get("cg_iterations", 10)):
            Ap = self._hessian_vector_product(loss_grads_D_c, p)
            alpha = rs_old / sum(torch.sum(p_g * ap_g) for p_g, ap_g in zip(p, Ap))

            x = [x_g + alpha * p_g for x_g, p_g in zip(x, p)]
            r = [r_g - alpha * ap_g for r_g, ap_g in zip(r, Ap)]

            rs_new = sum(torch.sum(r_g * r_g) for r_g in r)
            if torch.sqrt(rs_new) < 1e-10:
                break

            p = [r_g + (rs_new / rs_old) * p_g for r_g, p_g in zip(r, p)]
            rs_old = rs_new

        return x

    def _compute_parameter_change(self):
        """Computes the estimated parameter change to unlearn the data."""
        logger.info("Computing parameter change via influence functions...")

        # Grad_f(theta, D_r): Gradient w.r.t. the data to be removed
        grad_D_r = self._compute_loss_grad(self.model, self.data.edges_to_remove)

        # Approximate H^-1 * Grad_f(theta, D_r)
        param_change = self._conjugate_gradient_solve(grad_D_r)

        return param_change

    def _apply_parameter_change(self, parameter_change):
        """Applies the calculated parameter changes to a copy of the model."""
        unlearning_model = copy.deepcopy(self.model)
        with torch.no_grad():
            for p, change in zip(unlearning_model.parameters(), parameter_change):
                if p.requires_grad:
                    p.sub_(change)
        return unlearning_model

    def _repair_model(self, unlearning_model):
        """
        Repairs the unlearned model by fine-tuning it on a small subset of the
        remaining data to recover utility.
        """
        logger.info("Repairing the model after applying parameter changes...")
        repair_args = copy.deepcopy(self.args)
        repair_args["epoch"] = self.args.get("repair_epochs", 5)
        repair_args["lr"] = self.args.get("repair_lr", self.args["lr"] * 0.1)

        # Fine-tune on the remaining data
        repair_data = copy.deepcopy(self.data)
        repair_data.train_edge_index = self.data.train_edge_index_after_remove

        repaired_model = train_model(
            unlearning_model, repair_data, repair_args, self.device
        )
        return repaired_model

    def unlearn(self):
        """
        Executes the full L2UnRank unlearning process.
        """
        logger.info("Executing L2UnRank unlearning process...")
        start_time = time.time()

        # 1. Approximate parameter change using influence functions
        parameter_change = self._compute_parameter_change()

        # 2. Apply the parameter change to get an approximate unlearned model
        approx_unlearned_model = self._apply_parameter_change(parameter_change)

        # 3. Repair the model to recover utility
        final_unlearned_model = self._repair_model(approx_unlearned_model)

        unlearning_time = time.time() - start_time
        logger.info(
            f"L2UnRank unlearning complete. Time taken: {unlearning_time:.2f} seconds."
        )

        return final_unlearned_model, self.data.edges_to_remove, unlearning_time
