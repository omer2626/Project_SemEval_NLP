# # src/data_utils.py
# import pandas as pd
# from datasets import Dataset
# from transformers import AutoTokenizer

# def load_parquet_dataset(path, text_field="code", label_field="label"):
#     """
#     Load a parquet file and return a HuggingFace Dataset.
#     Expects columns: [code, label]
#     """
#     df = pd.read_parquet(path)
#     # Ensure the expected columns exist
#     if text_field not in df.columns or label_field not in df.columns:
#         raise ValueError(f"Parquet at {path} must contain columns: {text_field}, {label_field}")
#     return Dataset.from_pandas(df[[text_field, label_field]])

# class CodeDatasetPreprocessor:
#     def __init__(self, model_name_or_path, max_length=512, text_field="code"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
#         self.max_length = max_length
#         self.text_field = text_field

#     def tokenize_batch(self, examples):
#         return self.tokenizer(
#             examples[self.text_field],
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_length
#         )

#     def prepare(self, hf_dataset):
#         return hf_dataset.map(self.tokenize_batch, batched=True, remove_columns=[self.text_field])


# data_utils.py
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import pandas as pd
import os


def load_parquet_dataset(parquet_path, text_field="code", label_field="label", streaming=False):
    """
    Load a parquet file as a streaming Hugging Face dataset.

    This helper will try several common path resolutions so relative paths
    work whether you run scripts from the repo root or from `src/`.
    It returns the datasets IterableDataset produced by `load_dataset("parquet", ...)`
    (streaming).
    """
    # If an absolute path or relative path already exists, use it
    candidates = []
    # raw path as given
    candidates.append(parquet_path)
    # expanded user and absolute
    candidates.append(os.path.abspath(os.path.expanduser(parquet_path)))

    # try relative to this file (src/...) — useful when running from repo root
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(this_dir, parquet_path))

    # try project root (parent of src)
    project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
    candidates.append(os.path.join(project_root, parquet_path))

    # normalize and dedupe
    tried = []
    for p in candidates:
        try:
            p_norm = os.path.normpath(p)
        except Exception:
            p_norm = p
        if p_norm not in tried:
            tried.append(p_norm)

    existing = [p for p in tried if os.path.exists(p)]
    if not existing:
        error_msg = (
            f"Unable to find parquet file. Tried the following paths:\n"
            + "\n".join([f" - {p}" for p in tried])
            + "\nPlease provide an absolute path or run the script from the repository root."
        )
        raise FileNotFoundError(error_msg)

    selected = existing[0]
    # Use the selected existing path with datasets. Allow datasets to accept a string path.
    if streaming:
        ds = load_dataset("parquet", data_files=selected, split="train", streaming=True)
        return ds.map(lambda x: {text_field: x[text_field], label_field: x[label_field]})

    # Non-streaming: load into memory via pandas then convert to a Dataset
    df = pd.read_parquet(selected)
    if text_field not in df.columns or label_field not in df.columns:
        raise ValueError(f"Parquet at {selected} must contain columns: {text_field}, {label_field}")
    df = df[[text_field, label_field]].reset_index(drop=True)
    return Dataset.from_pandas(df)


class CodeDatasetPreprocessor:
    """
    Tokenizes code or text fields in Hugging Face datasets for transformer models.
    """

    def __init__(self, model_name_or_path, max_length=512, text_field="code"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.max_length = max_length
        self.text_field = text_field

    def tokenize_batch(self, examples):
        return self.tokenizer(
            examples[self.text_field],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

    def prepare(self, dataset):
        """
        Apply tokenization to a Hugging Face Dataset (non-streaming).
        """
        return dataset.map(
            self.tokenize_batch,
            batched=True,
            remove_columns=[self.text_field]
        )

# add one more file in src
# predict.py
# src/predict.py
# import torch
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
# from datasets import Dataset

# def main():
#     # --- Step 1: Paths ---
#     model_dir = "experiments/binary/codebert/checkpoint-375"  # use the latest checkpoint
#     test_path = "data/data/binary/test.parquet"
#     output_csv = "predictions.csv"

#     # --- Step 2: Load test data ---
#     print(f"📂 Loading test data from {test_path} ...")
#     test_df = pd.read_parquet(test_path)
#     test_df = test_df.reset_index(drop=True)
#     test_df["ID"] = test_df.index

#     # --- Step 3: Load tokenizer and model ---
#     print(f"🚀 Loading model and tokenizer from {model_dir} ...")
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)
#     model = AutoModelForSequenceClassification.from_pretrained(model_dir)

#     # --- Step 4: Tokenize test data ---
#     print("🔡 Tokenizing test data ...")
#     def preprocess(batch):
#         return tokenizer(batch["code"], truncation=True, padding="max_length", max_length=256)
    
#     test_ds = Dataset.from_pandas(test_df)
#     test_tokenized = test_ds.map(preprocess, batched=True)

#     # --- Step 5: Initialize Trainer for prediction ---
#     trainer = Trainer(model=model)

#     # --- Step 6: Predict ---
#     print("🤖 Running predictions ...")
#     preds = trainer.predict(test_tokenized)
#     logits = preds.predictions
#     y_pred = torch.argmax(torch.tensor(logits), dim=1).numpy()

#     # --- Step 7: Save results ---
#     submission = pd.DataFrame({
#         "ID": test_df["ID"],
#         "label": y_pred
#     })
#     submission.to_csv(output_csv, index=False)
#     print(f"✅ Predictions saved to {output_csv} (shape: {submission.shape})")

# if __name__ == "__main__":
#     main()