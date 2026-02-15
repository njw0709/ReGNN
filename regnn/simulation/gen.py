import numpy as np
import json


class GroundTruthFactory:
    def __init__(self, random_state=None):
        self.rng = np.random.RandomState(random_state)

    # ==========================================
    # 1. Soft Tree Generator
    # ==========================================
    def build_tree_blueprint(
        self,
        X,
        binary_indices=None,
        depth=3,
        interaction_strength=5.0,
        sharpness=2.0,
        oblique=False,
        n_split_features=2,
    ):
        """
        Build a soft decision tree blueprint.

        Args:
            X: Training data of shape (n_samples, n_features).
            binary_indices: Indices of binary features.
            depth: Maximum tree depth.
            interaction_strength: Range for leaf values.
            sharpness: Controls the steepness of the sigmoid splits.
                - 1.0 to 5.0: Very smooth, continuous transitions.
                - > 20.0: discrete step-function behavior.
            oblique: If True, each split uses a linear combination of
                ``n_split_features`` features instead of a single feature.
            n_split_features: Number of features per oblique split (ignored
                when ``oblique=False``).
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
                if oblique:
                    # Oblique split: linear combination of multiple features
                    k = min(n_split_features, n_features)
                    feat_indices = sorted(
                        self.rng.choice(n_features, k, replace=False).tolist()
                    )
                    weights = self.rng.randn(k).tolist()
                    # Normalise weights so scale is comparable across splits
                    w_norm = np.sqrt(sum(w ** 2 for w in weights))
                    if w_norm < 1e-8:
                        continue
                    weights = [w / w_norm for w in weights]

                    # Project data through the weight vector
                    node_X = np.sum(
                        X[np.ix_(indices, feat_indices)]
                        * np.array(weights)[None, :],
                        axis=1,
                    )
                else:
                    # Axis-aligned split: single feature
                    feat_indices = [int(self.rng.choice(n_features))]
                    weights = [1.0]
                    node_X = X[indices, feat_indices[0]]

                if len(np.unique(node_X)) < 2:
                    continue

                # Determine Threshold
                if not oblique and feat_indices[0] in bin_idx_set:
                    threshold = 0.5
                else:
                    lower, upper = np.percentile(node_X, 25), np.percentile(
                        node_X, 75
                    )
                    if upper - lower < 1e-6:
                        continue
                    threshold = float(self.rng.uniform(lower, upper))

                # Check split validity
                left_mask = node_X <= threshold
                if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
                    continue

                return {
                    "type": "split",
                    "feature_indices": feat_indices,
                    "weights": weights,
                    "threshold": threshold,
                    "sharpness": float(sharpness),
                    "left": build_node(indices[left_mask], current_depth + 1),
                    "right": build_node(indices[~left_mask], current_depth + 1),
                }

            val = self.rng.uniform(-interaction_strength, interaction_strength)
            return {"type": "leaf", "value": float(val)}

        return {
            "model_type": "soft_tree",
            "structure": build_node(np.arange(n_samples), 0),
        }

    # ==========================================
    # 2. Sparse Mixture (UPDATED: Per-Bump Features)
    # ==========================================
    def build_mixture_blueprint(
        self,
        n_features,
        binary_indices=None,
        n_components=5,
        sparsity=3,
        axis_aligned=True,
    ):
        """
        Creates 'n_components' bumps.
        Each bump selects its OWN distinct set of 'sparsity' features.

        Args:
            n_features: Total number of input features.
            binary_indices: Indices of binary features.
            n_components: Number of Gaussian bumps.
            sparsity: Number of features each bump operates on.
            axis_aligned: If True (default), bumps are axis-aligned spheres.
                If False, each bump gets a random linear transform ``L`` so
                the distance becomes ``||L @ (x_sub - centroid)||^2``,
                producing rotated ellipsoidal bumps.
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
                    centroid[i] = self.rng.uniform(-2.0, 2.0)

            # 3. Generate weight
            weight = self.rng.randn()

            comp = {
                "active_dims": active_dims.tolist(),
                "centroid": centroid.tolist(),
                "weight": float(weight),
            }

            # 4. Optional: random transform for non-axis-aligned bumps
            if not axis_aligned:
                # Random matrix L of shape (sparsity, sparsity).
                # Precision matrix = L^T @ L is guaranteed positive definite.
                # Scale entries so bumps don't collapse or explode.
                L = self.rng.randn(sparsity, sparsity)
                # Scale to have spectral norm ~ 1 (keeps bump widths reasonable)
                svs = np.linalg.svd(L, compute_uv=False)
                L = L / (svs[0] + 1e-8)
                comp["transform"] = L.tolist()

            components.append(comp)

        return {
            "model_type": "sparse_mixture",
            "components": components,
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
    # 4. Dense Generator (NEW)
    # ==========================================
    def build_dense_blueprint(self, n_features, hidden_dim=16, n_layers=2):
        """
        Creates a 'Dense' ground truth using a randomized Multi-Layer Perceptron (MLP).
        Every output depends on a linear combination of ALL inputs (propagated through layers).

        This mimics the 'Dense' regime where N approx d, or where many weak features combine.
        """
        layers = []

        # Input Dimension -> Hidden Dimension
        prev_dim = n_features

        for _ in range(n_layers):
            # Initialize weights (Xavier-like to keep variance stable)
            limit = np.sqrt(6 / (prev_dim + hidden_dim))
            W = self.rng.uniform(-limit, limit, size=(prev_dim, hidden_dim))
            b = self.rng.uniform(-0.1, 0.1, size=hidden_dim)

            layers.append({"W": W.tolist(), "b": b.tolist(), "activation": "tanh"})
            prev_dim = hidden_dim

        # Final Layer (Hidden -> Output 1)
        limit = np.sqrt(6 / (prev_dim + 1))
        W_out = self.rng.uniform(-limit, limit, size=(prev_dim, 1))
        b_out = self.rng.uniform(-0.1, 0.1, size=1)

        layers.append(
            {"W": W_out.tolist(), "b": b_out.tolist(), "activation": "linear"}
        )

        return {"model_type": "dense_mlp", "layers": layers}

    # ==========================================
    # 5. The Reconstructor
    # ==========================================
    @staticmethod
    def get_function_from_json(blueprint):

        if blueprint["model_type"] == "soft_tree":

            def soft_tree_predict_single(x, node):
                if node["type"] == "leaf":
                    return node["value"]

                # Compute the projection value for this split
                if "feature_indices" in node:
                    # New format: oblique or axis-aligned via weight vector
                    feat_val = sum(
                        w * x[idx]
                        for idx, w in zip(node["feature_indices"], node["weights"])
                    )
                else:
                    # Legacy format: single feature_index
                    feat_val = x[node["feature_index"]]

                threshold = node["threshold"]
                sharpness = node["sharpness"]
                prob_left = 1.0 / (1.0 + np.exp(-sharpness * (threshold - feat_val)))
                val_left = soft_tree_predict_single(x, node["left"])
                val_right = soft_tree_predict_single(x, node["right"])
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

        elif blueprint["model_type"] == "sparse_mixture":
            components = blueprint["components"]
            fast_components = []
            for c in components:
                active_dims = np.array(c["active_dims"])
                centroid = np.array(c["centroid"])
                weight = c["weight"]
                transform = np.array(c["transform"]) if "transform" in c else None
                fast_components.append((active_dims, centroid, weight, transform))

            def predict_mix(X):
                X = np.array(X, dtype=float)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                y = np.zeros(X.shape[0])
                for active_dims, centroid, weight, transform in fast_components:
                    X_sub = X[:, active_dims]
                    diff = X_sub - centroid  # (n_samples, sparsity)
                    if transform is not None:
                        # Rotated ellipsoidal bump: ||L @ diff||^2
                        diff = diff @ transform.T  # (n_samples, sparsity)
                    dist_sq = np.sum(diff ** 2, axis=1)
                    y += weight * np.exp(-0.5 * dist_sq)
                return y

            return predict_mix

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

        # --- DENSE MLP RECONSTRUCTION ---
        elif blueprint["model_type"] == "dense_mlp":
            layers_data = blueprint["layers"]
            # Pre-convert lists to numpy for speed
            np_layers = []
            for layer in layers_data:
                np_layers.append(
                    {
                        "W": np.array(layer["W"]),
                        "b": np.array(layer["b"]),
                        "act": layer["activation"],
                    }
                )

            def predict_dense(X):
                # Ensure input is 2D matrix
                H = np.array(X, dtype=float)
                if H.ndim == 1:
                    H = H.reshape(1, -1)

                for layer in np_layers:
                    # Linear Pass: H @ W + b
                    H = np.dot(H, layer["W"]) + layer["b"]

                    # Activation
                    if layer["act"] == "tanh":
                        H = np.tanh(H)
                    elif layer["act"] == "relu":
                        H = np.maximum(0, H)
                    # "linear" does nothing

                return H.flatten()  # Return 1D array of outcomes

            return predict_dense

        else:
            raise ValueError(f"Unknown model type: {blueprint.get('model_type')}")
