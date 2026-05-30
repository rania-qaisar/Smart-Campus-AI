"""
=============================================================================
Module  : ANN Priority Prediction Module
File    : ann_module.py
Course  : AL2002 – Artificial Intelligence Lab (Spring 2026)
Project : Smart Campus AI Decision Support and Automation System
=============================================================================
Description:
    This module implements two Artificial Neural Network models entirely
    from scratch using only Python built-ins and basic math (no NumPy/
    TensorFlow/PyTorch required).

    Model 1 — Perceptron (Binary Classifier)
        Input  : 7-element feature vector
        Output : urgent (1) or not_urgent (0)
        Use    : Baseline comparison model

    Model 2 — MLP — Multi-Layer Perceptron (Multiclass Classifier)
        Architecture : Input(7) → Hidden1(4) → Hidden2(3) → Output(4)
        Output classes : Low, Normal, High, Urgent
        Use    : Final operational priority predictor

    IMPORTANT: ANN predicts priority ONLY.
               It does NOT grant or deny permissions.
               Logic / KB is the gatekeeper for that.

    Feature order (fixed, must match preprocessing.py):
        [Role, RequestType, Severity, TimeSensitivity,
         CrowdLevel, Distance, Eligibility]

    Pipeline position: ROUTER  →  [ANN]  →  Logic/KB
=============================================================================
"""

import math
import random

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
BINARY_LABELS  = {0: "not_urgent", 1: "urgent"}
MULTI_LABELS   = {0: "Low", 1: "Normal", 2: "High", 3: "Urgent"}
PRIORITY_INDEX = {"Low": 0, "Normal": 1, "High": 2, "Urgent": 3}


# ===========================================================================
# Utility functions
# ===========================================================================

def sigmoid(x: float) -> float:
    """
    Sigmoid activation function: σ(x) = 1 / (1 + e^(-x)).
    Clipped to avoid overflow for extreme x values.

    Parameters:
        x (float): Input value.

    Returns:
        float: Output in range (0, 1).
    """
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    """
    ReLU activation: max(0, x).

    Parameters:
        x (float): Input value.

    Returns:
        float: max(0, x).
    """
    return max(0.0, x)


def softmax(vec: list) -> list:
    """
    Softmax activation for the output layer of the MLP.
    Converts raw scores to a probability distribution over classes.

    Parameters:
        vec (list): List of raw output scores (floats).

    Returns:
        list: Probability distribution (values sum to 1.0).
    """
    max_v = max(vec)
    exps  = [math.exp(v - max_v) for v in vec]
    total = sum(exps)
    return [e / total for e in exps]


def dot(w: list, x: list) -> float:
    """
    Dot product of two equal-length lists.

    Parameters:
        w (list): Weight vector.
        x (list): Input vector.

    Returns:
        float: Scalar dot product value.
    """
    return sum(wi * xi for wi, xi in zip(w, x))


def normalise_features(x: list) -> list:
    """
    Simple min-max normalisation of the 7-element feature vector.
    Feature ranges used:
        Role          : 0-2   → /2
        RequestType   : 0-4   → /4
        Severity      : 1-10  → /10
        TimeSensitivity: 1-10 → /10
        CrowdLevel    : 1-10  → /10
        Distance      : 0-6   → /6
        Eligibility   : 0-1   → /1

    Parameters:
        x (list): Raw 7-element feature vector.

    Returns:
        list: Normalised feature vector with all values in [0, 1].
    """
    divisors = [2.0, 4.0, 10.0, 10.0, 10.0, 6.0, 1.0]
    return [xi / d if d != 0 else 0.0 for xi, d in zip(x, divisors)]


# ===========================================================================
# PERCEPTRON — Binary Classifier
# ===========================================================================

class Perceptron:
    """
    Single-layer Perceptron for binary classification: urgent vs not_urgent.

    Architecture:
        Input layer (7 neurons) → weighted sum + bias → step activation → output

    Training method: Perceptron learning rule
        w_i ← w_i + η * (target - prediction) * x_i
        b   ← b   + η * (target - prediction)

    Attributes:
        n_inputs    (int)   : Number of input features (7).
        lr          (float) : Learning rate.
        epochs      (int)   : Training epochs.
        weights     (list)  : Learned weight vector.
        bias        (float) : Learned bias term.
        trained     (bool)  : Whether model has been trained.
    """

    def __init__(self, n_inputs: int = 7, lr: float = 0.1, epochs: int = 100):
        """
        Initialises Perceptron with small random weights.

        Parameters:
            n_inputs (int)  : Number of input features.
            lr       (float): Learning rate (default 0.1).
            epochs   (int)  : Number of training passes (default 100).
        """
        random.seed(42)
        self.n_inputs = n_inputs
        self.lr       = lr
        self.epochs   = epochs
        self.weights  = [random.uniform(-0.5, 0.5) for _ in range(n_inputs)]
        self.bias     = random.uniform(-0.5, 0.5)
        self.trained  = False

    def _step(self, z: float) -> int:
        """
        Step activation function: returns 1 if z >= 0 else 0.

        Parameters:
            z (float): Weighted sum input.

        Returns:
            int: Binary prediction (0 or 1).
        """
        return 1 if z >= 0 else 0

    def predict_raw(self, x: list) -> int:
        """
        Performs a single forward pass and returns the binary prediction.

        Parameters:
            x (list): Normalised 7-element feature vector.

        Returns:
            int: 0 (not_urgent) or 1 (urgent).
        """
        z = dot(self.weights, x) + self.bias
        return self._step(z)

    def train(self, training_data: list):
        """
        Trains the Perceptron using the Perceptron learning rule.

        Parameters:
            training_data (list): List of (feature_vector, label) tuples
                                  where label is 0 or 1.
        """
        for _ in range(self.epochs):
            for x, target in training_data:
                xn   = normalise_features(x)
                pred = self.predict_raw(xn)
                err  = target - pred
                self.weights = [w + self.lr * err * xi
                                for w, xi in zip(self.weights, xn)]
                self.bias   += self.lr * err
        self.trained = True

    def predict(self, x: list) -> dict:
        """
        Predicts urgency class for a single input and returns a result dict.

        Parameters:
            x (list): Raw 7-element feature vector (before normalisation).

        Returns:
            dict: {
                "binary_class": int,
                "binary_label": str,
                "confidence"  : float
            }
        """
        xn    = normalise_features(x)
        z     = dot(self.weights, xn) + self.bias
        pred  = self._step(z)
        # Approximate confidence via sigmoid of |z|
        conf  = round(sigmoid(abs(z)), 4)
        return {
            "binary_class": pred,
            "binary_label": BINARY_LABELS[pred],
            "confidence"  : conf
        }


# ===========================================================================
# MLP — Multi-Layer Perceptron (Multiclass)
# ===========================================================================

class MLP:
    """
    Multi-Layer Perceptron for 4-class priority classification:
        Low | Normal | High | Urgent

    Architecture:
        Input(7) → Hidden1(4, ReLU) → Hidden2(3, ReLU) → Output(4, Softmax)

    Training: Backpropagation with gradient descent.

    Attributes:
        lr          (float) : Learning rate.
        epochs      (int)   : Training epochs.
        w1, b1      (list)  : Weights and biases for layer 1.
        w2, b2      (list)  : Weights and biases for layer 2.
        w3, b3      (list)  : Weights and biases for output layer.
        trained     (bool)  : Whether model has been trained.
    """

    def __init__(self, lr: float = 0.05, epochs: int = 500):
        """
        Initialises MLP weights with Xavier-style uniform initialisation.

        Parameters:
            lr     (float): Learning rate (default 0.05).
            epochs (int)  : Training epochs (default 500).
        """
        random.seed(99)
        self.lr      = lr
        self.epochs  = epochs
        self.trained = False

        # Layer sizes
        self.in_size  = 7
        self.h1_size  = 4
        self.h2_size  = 3
        self.out_size = 4

        # Xavier initialisation: limit = sqrt(6 / (fan_in + fan_out))
        def xavier(fan_in, fan_out):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return [[random.uniform(-limit, limit) for _ in range(fan_in)]
                    for _ in range(fan_out)]

        self.w1 = xavier(self.in_size, self.h1_size)   # 4 x 7
        self.b1 = [0.0] * self.h1_size

        self.w2 = xavier(self.h1_size, self.h2_size)   # 3 x 4
        self.b2 = [0.0] * self.h2_size

        self.w3 = xavier(self.h2_size, self.out_size)  # 4 x 3
        self.b3 = [0.0] * self.out_size

    def _forward(self, x: list) -> tuple:
        """
        Forward pass through all layers.

        Parameters:
            x (list): Normalised input feature vector (length 7).

        Returns:
            tuple: (h1, h2, out_probs) where each is a list of activations.
        """
        # Hidden layer 1
        h1 = [relu(dot(self.w1[j], x) + self.b1[j]) for j in range(self.h1_size)]

        # Hidden layer 2
        h2 = [relu(dot(self.w2[j], h1) + self.b2[j]) for j in range(self.h2_size)]

        # Output layer with softmax
        raw_out = [dot(self.w3[j], h2) + self.b3[j] for j in range(self.out_size)]
        out     = softmax(raw_out)

        return h1, h2, out

    def _one_hot(self, label: int, size: int) -> list:
        """
        Creates a one-hot encoded target vector.

        Parameters:
            label (int): Class index.
            size  (int): Total number of classes.

        Returns:
            list: One-hot vector of given size.
        """
        v      = [0.0] * size
        v[label] = 1.0
        return v

    def train(self, training_data: list):
        """
        Trains the MLP using backpropagation and gradient descent.
        Uses cross-entropy loss with softmax output.

        Parameters:
            training_data (list): List of (feature_vector, label) tuples
                                  where label is 0-3 (Low/Normal/High/Urgent).
        """
        for epoch in range(self.epochs):
            for x_raw, target_label in training_data:
                x  = normalise_features(x_raw)
                h1, h2, out = self._forward(x)

                # One-hot target
                y = self._one_hot(target_label, self.out_size)

                # ── Output layer gradients (cross-entropy + softmax combined)
                # δ_out = out - y
                d_out = [out[k] - y[k] for k in range(self.out_size)]

                # ── Gradient for w3, b3
                dw3 = [[d_out[k] * h2[j] for j in range(self.h2_size)]
                        for k in range(self.out_size)]
                db3 = list(d_out)

                # ── Backprop through hidden layer 2
                # δ_h2[j] = Σ_k (δ_out[k] * w3[k][j]) * relu'(h2[j])
                d_h2 = []
                for j in range(self.h2_size):
                    s = sum(d_out[k] * self.w3[k][j] for k in range(self.out_size))
                    s *= (1.0 if h2[j] > 0 else 0.0)   # ReLU derivative
                    d_h2.append(s)

                dw2 = [[d_h2[j] * h1[i] for i in range(self.h1_size)]
                        for j in range(self.h2_size)]
                db2 = list(d_h2)

                # ── Backprop through hidden layer 1
                d_h1 = []
                for i in range(self.h1_size):
                    s = sum(d_h2[j] * self.w2[j][i] for j in range(self.h2_size))
                    s *= (1.0 if h1[i] > 0 else 0.0)
                    d_h1.append(s)

                dw1 = [[d_h1[i] * x[f] for f in range(self.in_size)]
                        for i in range(self.h1_size)]
                db1 = list(d_h1)

                # ── Weight updates (gradient descent)
                for k in range(self.out_size):
                    for j in range(self.h2_size):
                        self.w3[k][j] -= self.lr * dw3[k][j]
                    self.b3[k] -= self.lr * db3[k]

                for j in range(self.h2_size):
                    for i in range(self.h1_size):
                        self.w2[j][i] -= self.lr * dw2[j][i]
                    self.b2[j] -= self.lr * db2[j]

                for i in range(self.h1_size):
                    for f in range(self.in_size):
                        self.w1[i][f] -= self.lr * dw1[i][f]
                    self.b1[i] -= self.lr * db1[i]

        self.trained = True

    def predict(self, x: list) -> dict:
        """
        Predicts the 4-class priority label for a single input.

        Parameters:
            x (list): Raw 7-element feature vector (before normalisation).

        Returns:
            dict: {
                "class_index"   : int,
                "final_priority": str,   (Low/Normal/High/Urgent)
                "confidence"    : float,
                "probabilities" : dict   (all class probabilities)
            }
        """
        xn          = normalise_features(x)
        _, _, probs = self._forward(xn)
        idx         = probs.index(max(probs))
        return {
            "class_index"   : idx,
            "final_priority": MULTI_LABELS[idx],
            "confidence"    : round(probs[idx], 4),
            "probabilities" : {MULTI_LABELS[i]: round(probs[i], 4)
                               for i in range(self.out_size)}
        }


# ===========================================================================
# Synthetic training data generator
# ===========================================================================

def generate_training_data():
    """
    Generates a synthetic training dataset for both Perceptron and MLP.

    Features: [Role, RequestType, Severity, TimeSensitivity,
                CrowdLevel, Distance, Eligibility]

    Binary labels  : 0=not_urgent, 1=urgent
    Multiclass labels: 0=Low, 1=Normal, 2=High, 3=Urgent

    Heuristic rules used for labelling:
        Urgent   : severity >= 8  OR time_sensitivity >= 9
        High     : severity >= 6  OR time_sensitivity >= 7
        Normal   : severity >= 4  OR time_sensitivity >= 5
        Low      : everything else

    Returns:
        tuple: (binary_data, multi_data) — each is list of (x, label) tuples.
    """
    random.seed(7)
    binary_data = []
    multi_data  = []

    roles       = [0, 1, 2]
    req_types   = [0, 1, 2, 3, 4]
    eligible    = [0, 1]

    for _ in range(300):
        role    = random.choice(roles)
        rt      = random.choice(req_types)
        sev     = random.randint(1, 10)
        ts      = random.randint(1, 10)
        crowd   = random.randint(1, 10)
        dist    = random.randint(0, 6)
        elig    = random.choice(eligible)

        x = [role, rt, sev, ts, crowd, dist, elig]

        # Binary label
        b_label = 1 if (sev >= 7 or ts >= 8) else 0

        # Multiclass label
        if sev >= 8 or ts >= 9:
            m_label = 3   # Urgent
        elif sev >= 6 or ts >= 7:
            m_label = 2   # High
        elif sev >= 4 or ts >= 5:
            m_label = 1   # Normal
        else:
            m_label = 0   # Low

        binary_data.append((x, b_label))
        multi_data.append((x, m_label))

    return binary_data, multi_data


# ===========================================================================
# Module initialisation — train both models at import time
# ===========================================================================

def _init_models():
    """
    Initialises and trains both ANN models using synthetic training data.
    Called once when the module is imported.

    Returns:
        tuple: (perceptron_model, mlp_model)
    """
    binary_data, multi_data = generate_training_data()

    perceptron = Perceptron(n_inputs=7, lr=0.1, epochs=150)
    perceptron.train(binary_data)

    mlp = MLP(lr=0.05, epochs=600)
    mlp.train(multi_data)

    return perceptron, mlp


# Train models when module loads
print("[ANN] Training Perceptron and MLP models... ", end="", flush=True)
_perceptron_model, _mlp_model = _init_models()
print("Done.")


# ===========================================================================
# Public entry point for the pipeline
# ===========================================================================

def run_ann(request_obj: dict) -> dict:
    """
    Public entry point: runs both ANN models on the request's feature vector
    and returns a combined priority output dict.

    Uses the pre-trained Perceptron for binary urgency and the MLP for
    the final 4-class priority label. The MLP result is used operationally.

    Parameters:
        request_obj (dict): Preprocessed request object. Must contain
                            'ann_feature_vector' key.

    Returns:
        dict: {
            "binary_priority": str,   (urgent / not_urgent)
            "final_priority" : str,   (Low / Normal / High / Urgent)
            "confidence"     : float,
            "probabilities"  : dict,  (MLP class probabilities)
            "perceptron_conf": float  (Perceptron confidence)
        }
    """
    x = request_obj.get("ann_feature_vector", [0, 0, 5, 5, 5, 4, 1])

    # Run Perceptron (binary)
    p_result = _perceptron_model.predict(x)

    # Run MLP (multiclass)
    m_result = _mlp_model.predict(x)

    return {
        "binary_priority": p_result["binary_label"],
        "final_priority" : m_result["final_priority"],
        "confidence"     : m_result["confidence"],
        "probabilities"  : m_result["probabilities"],
        "perceptron_conf": p_result["confidence"]
    }
