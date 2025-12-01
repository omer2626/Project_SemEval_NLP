import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# === LABEL ORDER: must match training ===
LABELS = [
    "human",
    "DeepSeek-AI",
    "Qwen",
    "01-ai",
    "BigCode",
    "Gemma",
    "Phi",
    "Meta-LLaMA",
    "IBM-Granite",
    "Mistral",
    "OpenAI"
]

def load_model(model_dir, device=None):
    """
    Load trained model and tokenizer for multiclass classification.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, device


def predict_single(model, tokenizer, code_snippet, device):
    """
    Run prediction on a single code snippet.
    Returns predicted label and probabilities for all classes.
    """
    inputs = tokenizer(
        [code_snippet],
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()[0]
    pred = int(np.argmax(probs))
    return LABELS[pred], probs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multiclass Authorship Detection Predictor")
    parser.add_argument("--model_dir", required=True, help="Path to trained model directory")
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model_dir)

    print("🚀 Multiclass Model loaded successfully.")
    print("Type/paste your code snippet below. Press Enter twice to analyze. Ctrl+C to quit.\n")

    while True:
        try:
            print("=== Enter your code (finish with empty line) ===")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            code_snippet = "\n".join(lines)

            if not code_snippet.strip():
                continue

            # Optional: ask for actual label for evaluation
            actual_label = input("Enter the actual label (human/DeepSeek-AI/.../OpenAI), or press Enter to skip: ").strip()
            label, probs = predict_single(model, tokenizer, code_snippet, device)

            print("\n=== ANALYSIS ===")
            print(code_snippet)
            print("\n=== RESULTS ===")
            if actual_label:
                print(f"Actual Label    : {actual_label}")
            print(f"Predicted Label : {label}")
            print("\nClass Probabilities:")
            for i, l in enumerate(LABELS):
                print(f"  {l:12s}: {probs[i]:.4f}")

            if actual_label:
                if actual_label.lower() == label.lower():
                    print("\n✅ Prediction MATCHES the actual label.\n")
                else:
                    print("\n❌ Prediction DOES NOT match the actual label.\n")
            else:
                print()

        except KeyboardInterrupt:
            print("\nExiting.")
            break
