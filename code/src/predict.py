# # src/predict.py
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import numpy as np

# LABELS = ["human", "machine"]  # Assuming label 0 = human, 1 = machine

# def load_model(model_dir, device=None):
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)
#     model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
#     model.eval()
#     return tokenizer, model, device

# def predict_single(model, tokenizer, code_snippet, device):
#     inputs = tokenizer(
#         [code_snippet],
#         truncation=True,
#         padding=True,
#         return_tensors="pt",
#         max_length=512
#     ).to(device)
#     with torch.no_grad():
#         logits = model(**inputs).logits.cpu().numpy()
#     probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()[0]
#     pred = int(np.argmax(probs))
#     return LABELS[pred], probs

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_dir", required=True, help="Path to trained model directory")
#     args = parser.parse_args()

#     tokenizer, model, device = load_model(args.model_dir)

#     print("🚀 Model loaded. Type/paste your code below. Press Enter twice to analyze. Ctrl+C to quit.\n")
#     while True:
#         try:
#             # Multi-line input until empty line
#             print("=== Enter your code (finish with empty line) ===")
#             lines = []
#             while True:
#                 line = input()
#                 if line.strip() == "":
#                     break
#                 lines.append(line)
#             code_snippet = "\n".join(lines)

#             if not code_snippet.strip():
#                 continue

#             # Ask user for actual label
#             actual_label = input("Enter the actual label (human/machine): ").strip().lower()
#             while actual_label not in LABELS:
#                 actual_label = input("❗ Invalid label. Please enter 'human' or 'machine': ").strip().lower()

#             # Run prediction
#             predicted_label, probs = predict_single(model, tokenizer, code_snippet, device)

#             # Display results
#             print("\n=== ANALYSIS ===")
#             print(code_snippet)
#             print("\n=== RESULTS ===")
#             print(f"Actual Label    : {actual_label.upper()}")
#             print(f"Predicted Label : {predicted_label.upper()}")
#             print(f"Probabilities   : human={probs[0]:.4f}, machine={probs[1]:.4f}")

#             if actual_label == predicted_label:
#                 print("✅ Prediction MATCHES the actual label.\n")
#             else:
#                 print("❌ Prediction DOES NOT match the actual label.\n")

#         except KeyboardInterrupt:
#             print("\nExiting.")
#             break


import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset

def main():
    # --- Step 1: Paths ---
    model_dir = "experiments/binary/codebert/checkpoint-375"  # use the latest checkpoint
    test_path = "data/data/binary/test.parquet"
    output_csv = "predictions.csv"

    # --- Step 2: Load test data ---
    print(f"📂 Loading test data from {test_path} ...")
    test_df = pd.read_parquet(test_path)
    test_df = test_df.reset_index(drop=True)
    test_df["ID"] = test_df.index

    # --- Step 3: Load tokenizer and model ---
    print(f"🚀 Loading model and tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # --- Step 4: Tokenize test data ---
    print("🔡 Tokenizing test data ...")
    def preprocess(batch):
        return tokenizer(batch["code"], truncation=True, padding="max_length", max_length=256)
    
    test_ds = Dataset.from_pandas(test_df)
    test_tokenized = test_ds.map(preprocess, batched=True)

    # --- Step 5: Initialize Trainer for prediction ---
    trainer = Trainer(model=model)

    # --- Step 6: Predict ---
    print("🤖 Running predictions ...")
    preds = trainer.predict(test_tokenized)
    logits = preds.predictions
    y_pred = torch.argmax(torch.tensor(logits), dim=1).numpy()

    # --- Step 7: Save results ---
    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "label": y_pred
    })
    submission.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv} (shape: {submission.shape})")

if __name__ == "__main__":
    main()