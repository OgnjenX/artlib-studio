"""
Minimal Fuzzy ART Example
==========================

A simple example demonstrating:
- FuzzyART initialization
- Training on synthetic data
- Making predictions
- Printing results

This serves as a verification that AdaptiveResonanceLib works correctly
in the development environment.
"""

from __future__ import annotations

import numpy as np

from artlib import FuzzyART


def main() -> None:
    """Run minimal Fuzzy ART example."""
    print("=" * 60)
    print("Minimal Fuzzy ART Example - Development Verification")
    print("=" * 60)

    # Create synthetic dataset
    # Two clusters: [~0.1, ~0.1] and [~0.8, ~0.8]
    X = np.array(
        [
            [0.10, 0.15],
            [0.12, 0.18],
            [0.80, 0.82],
            [0.78, 0.79],
        ],
        dtype=float,
    )

    print(f"\nDataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    # Initialize Fuzzy ART model
    # rho: vigilance parameter (controls category specificity)
    # alpha: choice parameter (controls activation)
    # beta: learning rate
    model = FuzzyART(rho=0.7, alpha=0.001, beta=1.0)

    print("\nFuzzy ART Configuration:")
    print(f"  - Vigilance (rho): {model.rho}")
    print(f"  - Choice (alpha): {model.alpha}")
    print(f"  - Learning rate (beta): {model.beta}")

    # Prepare data (normalize to [0, 1])
    X_prepared = model.prepare_data(X)

    # Train the model
    print("\nTraining the model...")
    model.fit(X_prepared)

    # Get predictions
    category_assignments = model.predict(X_prepared)

    # Display results
    print(f"\nResults:")
    print(f"  - Number of categories created: {model.n_clusters}")
    print(f"  - Category assignments: {category_assignments.tolist()}")

    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()