# Synthetic Communications Dataset

This directory contains synthetic communication messages generated for training and testing Complexio's communication classification systems. The dataset provides representative examples of both legitimate business communications and content that should be filtered from enterprise communication channels.

## Directory Structure

### `allow_and_deny_samples/`

Training data organised for the Bayesian communication classifier. This directory contains labelled message examples structured in the format required by the `BayesianTrainer` class from Complexio's communication classifier module.

The dataset includes 90 subcategories across 13 content categories: three allow categories (maritime shipping, marine insurance, general business) and ten deny categories (personal matters, HR relations, security risks, confidential strategy, legal compliance, inappropriate material, entertainment, spam, technology misuse, and time wasting).

Each message has been manually categorised by action (allow or deny), category, and subcategory to provide supervised training data for the multinomial Naive Bayes classifier.

See `allow_and_deny_samples/README.md` for detailed structure and usage instructions.

## Related Systems

This training data is consumed by:
- `lib/core/src/core/communication_classifier/` - Bayesian, regex, and hybrid classifiers
- `lib/core/src/core/communication_classifier/bayesian/default_classifier/utils/retrain_bayesian.py` - Retraining utility

The default classifier model included with Complexio has been trained on this dataset.
