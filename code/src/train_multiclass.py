# src/train_multiclass.py
import os
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from data_utils import load_parquet_dataset, CodeDatasetPreprocessor
from models import get_sequence_classification_model
from metrics import classification_metrics
import numpy as np

def subsample_dataset(dataset, max_size=6000, seed=42):
    """Subsample Hugging Face dataset or pandas DataFrame to max_size."""
    if len(dataset) > max_size:
        try:
            # Hugging Face datasets have shuffle & select
            dataset = dataset.shuffle(seed=seed).select(range(max_size))
        except AttributeError:
            # Fallback: assume pandas DataFrame
            dataset = dataset.sample(n=max_size, random_state=seed)
    return dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return classification_metrics(np.array(logits), np.array(labels))

def main():
    # -- config: change model_name & num_labels to suit your dataset --
    model_name = "microsoft/codebert-base"  # change if you prefer CodeT5 etc.
    train_path = "data/multiclass/train.parquet"
    val_path = "data/multiclass/val.parquet"
    out_dir = "experiments/multiclass/codebert"
    num_labels = 11   # adjust according to your dataset
    epochs = 5
    batch_size = 8
    lr = 3e-5

    os.makedirs(out_dir, exist_ok=True)

    train_ds = load_parquet_dataset(train_path)
    val_ds = load_parquet_dataset(val_path)
    train_ds = subsample_dataset(train_ds, max_size=3000)
    val_ds = subsample_dataset(val_ds, max_size=3000)
    preproc = CodeDatasetPreprocessor(model_name, max_length=256)
    train_tok = preproc.prepare(train_ds)
    val_tok = preproc.prepare(val_ds)

    model = get_sequence_classification_model(model_name, num_labels=num_labels)

    args = TrainingArguments(
    output_dir=out_dir,
    evaluation_strategy="epoch",   # run eval after each epoch
    save_strategy="epoch",         # save checkpoint after each epoch
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,            # can increase to 5–10 for better results
    weight_decay=0.01,
    logging_steps=50,
    fp16=True,                     # if on GPU
    save_total_limit=2,            # keep only last 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

    data_collator = DataCollatorWithPadding(tokenizer=preproc.tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=preproc.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(out_dir)
    preproc.tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
