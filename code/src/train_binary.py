# # src/train_binary.py
# import os
# import numpy as np
# from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
# from data_utils import load_parquet_dataset, CodeDatasetPreprocessor
# from models import get_sequence_classification_model
# from metrics import classification_metrics

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     return classification_metrics(np.array(logits), np.array(labels))

# def subsample_dataset(dataset, max_size=6000, seed=42):
#     """Subsample Hugging Face dataset or pandas DataFrame to max_size."""
#     if len(dataset) > max_size:
#         try:
#             # Hugging Face datasets have shuffle & select
#             dataset = dataset.shuffle(seed=seed).select(range(max_size))
#         except AttributeError:
#             # Fallback: assume pandas DataFrame
#             dataset = dataset.sample(n=max_size, random_state=seed)
#     return dataset


# def main():
#     # -- config: change paths / model / hyperparams as needed --
#     model_name = "microsoft/codebert-base"
#     train_path = "data/data/binary/train.parquet"
#     val_path = "data/data/binary/val.parquet"
#     out_dir = "experiments/binary/codebert"
#     num_labels = 2
#     epochs = 5
#     batch_size = 8
#     lr = 2e-5

#     os.makedirs(out_dir, exist_ok=True)

#     # Load datasets
#     train_ds = load_parquet_dataset(train_path)
#     val_ds = load_parquet_dataset(val_path)

#     # Subsample to 6000 examples each
#     train_ds = subsample_dataset(train_ds, max_size=3000)
#     val_ds = subsample_dataset(val_ds, max_size=3000)
#     # Preprocess / tokenize
#     preproc = CodeDatasetPreprocessor(model_name, max_length=256)
#     train_tok = preproc.prepare(train_ds)
#     val_tok = preproc.prepare(val_ds)

#     # Load model
#     model = get_sequence_classification_model(model_name, num_labels=num_labels)

#     # Training args
#     args = TrainingArguments(
#     output_dir=out_dir,
#     evaluation_strategy="epoch",   # run eval after each epoch
#     save_strategy="epoch",         # save checkpoint after each epoch
#     learning_rate=lr,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=epochs,            # can increase to 5–10 for better results
#     weight_decay=0.01,
#     logging_steps=50,
#     fp16=True,                     # if on GPU
#     save_total_limit=2,            # keep only last 2 checkpoints
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
# )
#     data_collator = DataCollatorWithPadding(tokenizer=preproc.tokenizer)

#     # Trainer
#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=train_tok,
#         eval_dataset=val_tok,
#         tokenizer=preproc.tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )

#     # Train and save
#     trainer.train()
#     trainer.save_model(out_dir)
#     preproc.tokenizer.save_pretrained(out_dir)


# if __name__ == "__main__":
#     main()


import os
import shutil
import numpy as np
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from data_utils import load_parquet_dataset, CodeDatasetPreprocessor
from models import get_sequence_classification_model
from metrics import classification_metrics
from datasets import Dataset
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return classification_metrics(np.array(logits), np.array(labels))


def take_samples(streaming_dataset, max_size=3000):
    """
    Take first `max_size` examples from a streaming dataset
    and convert to regular Hugging Face Dataset.
    """
    print(f"📦 Sampling first {max_size} examples from streaming dataset...")
    samples = []
    for i, example in enumerate(streaming_dataset):
        samples.append(example)
        if i + 1 >= max_size:
            break
    return Dataset.from_list(samples)


def choose_output_dir(base_dir="experiments/binary/codebert"):
    """
    Check available disk space and choose best directory for model saving.
    """
    try:
        total, used, free = shutil.disk_usage("C:/")
        free_gb = free / (1024 ** 3)
        if free_gb < 10:
            if os.path.exists("D:/"):
                alt_dir = "D:/models/binary/codebert"
                print(f"⚠️ Low disk space on C: ({free_gb:.2f} GB free). Redirecting output to {alt_dir}")
                return alt_dir
            else:
                alt_dir = "C:/models_temp/binary/codebert"
                print(f"⚠️ Low disk space on C: ({free_gb:.2f} GB free). Using {alt_dir} instead.")
                return alt_dir
        else:
            return base_dir
    except Exception as e:
        print(f"⚠️ Could not check disk space ({e}), using default output dir.")
        return base_dir


def main():
    # --- Config ---
    model_name = "microsoft/codebert-base"
    train_path = "data/data/binary/train.parquet"
    val_path = "data/data/binary/val.parquet"
    out_dir = choose_output_dir()
    num_labels = 2
    epochs = 3
    batch_size = 8
    lr = 2e-5

    os.makedirs(out_dir, exist_ok=True)

    print("📂 Loading training and validation data (streaming)...")
    train_stream = load_parquet_dataset(train_path)
    val_stream = load_parquet_dataset(val_path)

    # Convert limited samples to fit in memory
    train_ds = take_samples(train_stream, max_size=3000)
    val_ds = take_samples(val_stream, max_size=1000)

    # Preprocess / tokenize
    print("🔡 Tokenizing datasets...")
    preproc = CodeDatasetPreprocessor(model_name, max_length=256, text_field="code")
    train_tok = preproc.prepare(train_ds)
    val_tok = preproc.prepare(val_ds)

    # Load model
    print("🚀 Initializing model...")
    model = get_sequence_classification_model(model_name, num_labels=num_labels)

    # Training arguments
    print("⚙️ Setting up training arguments...")
    # Build kwargs and pass only supported args to TrainingArguments so this
    # script remains compatible with different transformers versions.
    import inspect

    training_kwargs = dict(
        output_dir=out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        fp16=False,  # must stay False on CPU
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Filter kwargs by what TrainingArguments.__init__ accepts
    try:
        sig = inspect.signature(TrainingArguments.__init__)
        valid_keys = set(sig.parameters.keys())
        # remove 'self' if present
        valid_keys.discard("self")
        filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in valid_keys}
        args = TrainingArguments(**filtered_kwargs)
    except Exception:
        # Fallback: try to construct with a minimal safe set
        minimal = dict(output_dir=out_dir, learning_rate=lr, per_device_train_batch_size=batch_size,
                       per_device_eval_batch_size=batch_size, num_train_epochs=epochs)
        args = TrainingArguments(**minimal)

    data_collator = DataCollatorWithPadding(tokenizer=preproc.tokenizer)

    # Trainer
    print(f"🏋️ Starting training (output dir: {out_dir})...")
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

    # Save model
    print("💾 Saving model and tokenizer...")
    trainer.save_model(out_dir)
    preproc.tokenizer.save_pretrained(out_dir)
    print(f"✅ Training complete! Model saved to: {out_dir}")


if __name__ == "__main__":
    main()