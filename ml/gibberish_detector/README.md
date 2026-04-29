# Gibberish Detector

A tiny (~1.3M param) character-trigram MLP that classifies search queries as
real (code-related) vs gibberish. Used by Grafyx to filter nonsense queries.

## Pipeline

1. `generate_data.py` — Generate 100K labeled training examples
2. `train.py` — Train the model (run on Colab with GPU)
3. `evaluate.py` — Evaluate accuracy, show confusion matrix
4. Trained weights saved to `model/gibberish_weights.npz`

## Inference

The trained model runs with pure numpy — no PyTorch needed at inference time.
See `inference.py` for the standalone inference class.
