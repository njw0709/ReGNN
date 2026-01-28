import numpy as np
import json


class GroundTruthFactory:
    def __init__(self, random_state=None):
        self.rng = np.random.RandomState(random_state)

    # ==========================================
    # 1. Soft Tree Generator
    # ==========================================
    def build_tree_blueprint(
        self, X, binary_indices=None, depth=3, interaction_strength=5.0, sharpness=2.0
    ):
        """
        sharpness: Controls the 'steepness' of the splits.
                   - 1.0 to 5.0: Very smooth, continuous transitions.
                   - > 20.0: discrete step-function behavior.
        """
        n_samples, n_features = X.shape
        bin_idx_set = set(binary_indices) if binary_indices else set()

        def build_node(indices, current_depth):
            # Base Case
            if current_depth >= depth or len(indices) < 10:
                val = self.rng.uniform(-interaction_strength, interaction_strength)
                return {"type": "leaf", "value": float(val)}

            # Find Split
            for _ in range(5):
                feat_idx = int(self.rng.choice(n_features))
                node_X = X[indices, feat_idx]
                if len(np.unique(node_X)) < 2:
                    continue

                # Determine Threshold
                if feat_idx in bin_idx_set:
                    threshold = 0.5
                else:
                    lower, upper = np.percentile(node_X, 25), np.percentile(node_X, 75)
                    if upper - lower < 1e-6:
                        continue
                    threshold = self.rng.uniform(lower, upper)

                # Check split validity (using Hard logic for construction to ensure balance)
                left_mask = node_X <= threshold
                if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
                    continue

                return {
                    "type": "split",
                    "feature_index": feat_idx,
                    "threshold": float(threshold),
                    "sharpness": float(sharpness),  # Store sharpness in the node
                    "left": build_node(indices[left_mask], current_depth + 1),
                    "right": build_node(indices[~left_mask], current_depth + 1),
                }

            val = self.rng.uniform(-interaction_strength, interaction_strength)
            return {"type": "leaf", "value": float(val)}

        return {
            "model_type": "soft_tree",  # Changed type name
            "structure": build_node(np.arange(n_samples), 0),
        }

    # ==========================================
    # 2. Sparse Mixture (UPDATED: Per-Bump Features)
    # ==========================================
    def build_mixture_blueprint(
        self, n_features, binary_indices=None, n_components=5, sparsity=3
    ):
        """
        Creates 'n_components' bumps.
        Each bump selects its OWN distinct set of 'sparsity' features.
        """
        bin_idx_set = set(binary_indices) if binary_indices else set()

        components = []

        for _ in range(n_components):
            # 1. Choose features specific to THIS bump
            active_dims = self.rng.choice(n_features, sparsity, replace=False)

            # 2. Generate centroid for these specific dims
            centroid = np.zeros(sparsity)
            for i, dim in enumerate(active_dims):
                if dim in bin_idx_set:
                    centroid[i] = self.rng.choice([0.0, 1.0])
                else:
                    # Continuous centroid in range [-2, 2]
                    centroid[i] = self.rng.uniform(-2.0, 2.0)

            # 3. Generate weight
            weight = self.rng.randn()

            components.append(
                {
                    "active_dims": active_dims.tolist(),
                    "centroid": centroid.tolist(),
                    "weight": float(weight),
                }
            )

        return {
            "model_type": "sparse_mixture",
            "components": components,  # List of independent bumps
        }

    # ==========================================
    # 3. Polynomial Generator (Unchanged)
    # ==========================================
    def build_polynomial_blueprint(self, n_features):
        def get_weight():
            return self.rng.uniform(0.05, 0.3) * self.rng.choice([-1, 1])

        poly_terms = []
        # Linear
        for _ in range(3):
            poly_terms.append(
                {
                    "degree": 1,
                    "indices": [int(self.rng.choice(n_features))],
                    "weight": float(get_weight()),
                }
            )
        # Higher Order
        for degree in [2, 3, 4]:
            for _ in range(3):
                indices = self.rng.choice(n_features, size=degree, replace=True)
                poly_terms.append(
                    {
                        "degree": degree,
                        "indices": [int(i) for i in indices],
                        "weight": float(get_weight()),
                    }
                )

        return {"model_type": "polynomial", "terms": poly_terms}

    # ==========================================
    # 4. The Reconstructor (With Soft Logic)
    # ==========================================
    @staticmethod
    def get_function_from_json(blueprint):

        # --- SOFT TREE LOGIC ---
        if blueprint["model_type"] == "soft_tree":

            def soft_tree_predict_single(x, node):
                if node["type"] == "leaf":
                    return node["value"]

                # Retrieve parameters
                feat_val = x[node["feature_index"]]
                threshold = node["threshold"]
                sharpness = node["sharpness"]

                # --- The Magic: Sigmoid Gating ---
                # This calculates the probability of going LEFT
                # If feat_val << threshold, exponent is large positive -> prob close to 1
                # If feat_val >> threshold, exponent is large negative -> prob close to 0
                prob_left = 1.0 / (1.0 + np.exp(-sharpness * (threshold - feat_val)))

                # Recursively get values from children
                val_left = soft_tree_predict_single(x, node["left"])
                val_right = soft_tree_predict_single(x, node["right"])

                # Weighted average based on probability
                return (prob_left * val_left) + ((1.0 - prob_left) * val_right)

            return lambda X: np.array(
                [
                    soft_tree_predict_single(x, blueprint["structure"])
                    for x in (
                        np.array(X, dtype=float)
                        if X.ndim > 1
                        else np.array(X, dtype=float).reshape(1, -1)
                    )
                ]
            )

        # --- Mixture Logic ---
        elif blueprint["model_type"] == "sparse_mixture":
            components = blueprint["components"]
            fast_components = [
                (np.array(c["active_dims"]), np.array(c["centroid"]), c["weight"])
                for c in components
            ]

            def predict_mix(X):
                X = np.array(X, dtype=float)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                y = np.zeros(X.shape[0])
                for active_dims, centroid, weight in fast_components:
                    X_sub = X[:, active_dims]
                    dist_sq = np.sum((X_sub - centroid) ** 2, axis=1)
                    y += weight * np.exp(-0.5 * dist_sq)
                return y

            return predict_mix

        # --- Polynomial Logic ---
        elif blueprint["model_type"] == "polynomial":
            terms = blueprint["terms"]

            def predict_poly(X):
                X = np.array(X, dtype=float)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                y = np.zeros(X.shape[0])
                for term in terms:
                    y += term["weight"] * np.prod(X[:, term["indices"]], axis=1)
                return y

            return predict_poly

        else:
            raise ValueError(f"Unknown model type: {blueprint.get('model_type')}")
