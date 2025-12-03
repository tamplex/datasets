# Complexio Datasets

Training and test data for Complexio's machine learning systems. This repository is separate from the main codebase to facilitate data versioning, reduce repository size, and enable controlled access to training data that should not be distributed to customers.

## Structure

### `synthetic/`

Synthetically generated datasets for training and testing:

- **`communications/`** - Communication classification training data
  - `allow_and_deny_samples/` - 90 subcategories (24 allow, 66 deny) for Bayesian classifier training
  - Organised as `{action}/{category}/{subcategory}.txt` for use with `BayesianTrainer`

## Usage

Point Complexio's training utilities to this repository:

```python
# In retrain_bayesian.py
training_dir = "path-to/datasets/synthetic/communications/allow_and_deny_samples"
```

Or load data programmatically:

```python
from core.communication_classifier.bayesian.trainer import BayesianTrainer

trainer = BayesianTrainer()
# Load messages from this repository's structure
```

## Related Repositories

- **complexio/complexio** - Main Complexio codebase including classifiers, trainers, and applications
- Communication classifier: `lib/core/src/core/communication_classifier/`
