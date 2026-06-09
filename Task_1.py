

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import gradio as gr


MODEL_NAME   = "bert-base-uncased"
MAX_LENGTH   = 128
BATCH_SIZE   = 16
EPOCHS       = 3
OUTPUT_DIR   = "./bert_ag_news"

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


print("Loading AG News dataset...")
dataset = load_dataset("ag_news")

print(f"Train samples : {len(dataset['train'])}")
print(f"Test  samples : {len(dataset['test'])}")
print(f"Sample record : {dataset['train'][0]}")


tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

print("\nTokenising dataset…")
tokenized = dataset.map(tokenize, batched=True)

tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized["train"].shuffle(seed=42).select(range(10_000))
eval_dataset  = tokenized["test"].shuffle(seed=42).select(range(2_000))


model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_NAMES),
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="weighted")
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=100,
    seed=42,
    report_to="none",  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting fine-tuning…")
trainer.train()

print("\nEvaluating on test set…")
results = trainer.evaluate()
print(f"  Accuracy : {results['eval_accuracy']}")
print(f"  F1 Score : {results['eval_f1']}")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to '{OUTPUT_DIR}'")

loaded_model     = BertForSequenceClassification.from_pretrained(OUTPUT_DIR)
loaded_tokenizer = BertTokenizerFast.from_pretrained(OUTPUT_DIR)
loaded_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model.to(device)

def predict(headline: str) -> dict:
    inputs = loaded_tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = loaded_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    return {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="News Headline",
        placeholder="e.g. NASA launches new Mars mission",
    ),
    outputs=gr.Label(num_top_classes=4, label="Category Probabilities"),
    title="📰 AG News Classifier (BERT)",
    description="Fine-tuned bert-base-uncased on AG News. Enter any headline to classify it.",
    examples=[
        ["Stock markets rally after Fed signals rate pause"],
        ["Champions League final ends in dramatic penalty shootout"],
        ["Scientists discover new exoplanet in habitable zone"],
        ["G7 leaders meet to discuss climate change policy"],
    ],
)

if __name__ == "__main__":
    demo.launch()