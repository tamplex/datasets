# Bayesian Classifier Training Data

This directory contains labelled training data for Complexio's Bayesian communication classifier. Messages are organised by action (allow or deny), category, and subcategory in the format required by the `BayesianTrainer` class.

## Directory Structure

```
allow_and_deny_samples/
├── allow/
│   ├── business_legitimate_general/
│   │   ├── client_relations.txt
│   │   ├── meetings_schedules.txt
│   │   └── ... (6 subcategories)
│   ├── marine_insurance/
│   │   ├── claims_processing.txt
│   │   ├── policy_administration.txt
│   │   └── ... (8 subcategories)
│   └── maritime_shipping_legitimate/
│       ├── cargo_logistics.txt
│       ├── vessel_operations.txt
│       └── ... (10 subcategories)
└── deny/
    ├── confidential_strategy/
    ├── entertainment_media/
    ├── hr_employee_relations/
    ├── inappropriate_material/
    ├── legal_compliance/
    ├── personal_matters/
    ├── personal_technology/
    ├── security_risks/
    ├── spam_marketing/
    └── time_wasting_content/
```

## File Format

Each `.txt` file represents a subcategory and contains training messages (one message per line). Messages may span multiple logical lines but are separated by blank lines or line breaks. For example:

```
Ship departure scheduled for 15:00 UTC The MV Ocean Glory is set to depart from Port Elizabeth...
ETD update for container vessel Albatross The Albatross will be casting off from Berth 7...
Vessel movement alert – tanker departure imminent The tanker Searise is scheduled to leave...
```

## Dataset Statistics

- **Total subcategories**: 90 (24 allow, 66 deny)
- **Allow categories**: 3 (business_legitimate_general, marine_insurance, maritime_shipping_legitimate)
- **Deny categories**: 10 (confidential_strategy, entertainment_media, hr_employee_relations, inappropriate_material, legal_compliance, personal_matters, personal_technology, security_risks, spam_marketing, time_wasting_content)
- **Messages per subcategory**: Varies (typically 100-1000 messages)

## Using This Data for Training

### With BayesianTrainer

The `BayesianTrainer` class from `lib/core/src/core/communication_classifier/bayesian/trainer.py` expects this exact directory structure:

```python
from core.communication_classifier.bayesian.trainer import BayesianTrainer

trainer = BayesianTrainer(use_oversampling=True)

# Add samples programmatically
for message in load_messages_from_files():
    trainer.add_training_sample(
        text=message["text"],
        action=message["action"],  # "allow" or "deny"
        category=message["category"],  # e.g., "maritime_shipping_legitimate"
        subcategory=message["subcategory"]  # e.g., "vessel_operations"
    )

# Train and save model
trainer.train_from_samples(
    model_output_path="communication_classifier_model.pkl",
    perform_cv=True,
    cv_folds=5
)
```

### With retrain_bayesian.py Utility

The retraining utility (`lib/core/src/core/communication_classifier/bayesian/default_classifier/utils/retrain_bayesian.py`) automatically loads training data from directories matching this structure. Point the script to this directory by updating the `training_dir` variable (line 125):

```python
training_dir = "/Users/tam/Repositories/datasets/synthetic/communications/allow_and_deny_samples"
```

The utility will:
1. Load all messages from `allow/{category}/{subcategory}.txt` and `deny/{category}/{subcategory}.txt`
2. Train a new Bayesian model with cross-validation
3. Test the model against validation data
4. Merge SME tuning weights (preserving domain expert customisations)
5. Update the production model file

## Training Process

The Bayesian classifier uses:
- **TF-IDF vectorisation** (max 20,000 features, English stopwords)
- **Multinomial Naive Bayes** classification
- **Random oversampling** for imbalanced datasets (optional)
- **Stratified k-fold cross-validation** (default 5 folds)

Training produces:
- `communication_classifier_model.pkl` - Serialised model (classifier, vectoriser, label encoder, action mappings)
- `classification_tuning.json` - SME weight configuration (all categories default to 1.0)

## SME Weight Tuning

After training, domain experts can adjust category sensitivity by editing `classification_tuning.json`:

```json
{
  "business_legitimate_general:client_relations": 1.0,
  "maritime_shipping_legitimate:vessel_operations": 1.2,
  "personal_matters:family_relationships": 0.8
}
```

Weights above 1.0 increase sensitivity (more likely to classify as that category); weights below 1.0 decrease sensitivity. The `merge_classification_tuning()` function in `retrain_bayesian.py` preserves these customisations when retraining, ensuring SMEs never lose their tuning work.

## Related Documentation

- **Communication classifier module**: `lib/core/src/core/communication_classifier/README.md`
- **Bayesian trainer**: `lib/core/src/core/communication_classifier/bayesian/trainer.py`
- **Retraining utility**: `lib/core/src/core/communication_classifier/bayesian/default_classifier/utils/retrain_bayesian.py`
- **Test data**: `lib/core/tests/communication_classifier/bayesian_default_classifier_test_data.tsv`

## Testing Trained Models

After training, validate the model using the test suite:

```bash
uv run pytest lib/core/tests/communication_classifier/ -v
```

The test suite includes ~200 synthesised messages covering all subcategories, with expected action, category, and subcategory classifications.
