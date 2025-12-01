# src/models.py
from transformers import AutoModelForSequenceClassification, AutoConfig

def get_sequence_classification_model(name_or_path, num_labels, from_pretrained=True):
    if from_pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(name_or_path, num_labels=num_labels)
    else:
        config = AutoConfig.from_pretrained(name_or_path, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_config(config)
    return model
