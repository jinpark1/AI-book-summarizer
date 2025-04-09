from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
import torch
import optuna

# Check GPU if GPU not available choose CPU.
# Highly recommended that GPU is used for performance.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Load Model and Tokenizer
# Use t5-base or t5-small
# (t5-base more accurate, but takes longer)
# (t5-small fast, but less accurate)
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Load dataset to train the model, booksum contains books and their summaries.
dataset = load_dataset("kmfoda/booksum")

# Select a subset of data to train
# Use 500 examples to train and 100 for evaluation.
train_data = dataset["train"].select(range(100))
eval_data = dataset["validation"].select(range(20))

def preprocess_data(batch):
    # Convert content and summaries to strings
    contents = [str(text) if text else "" for text in batch.get("content", [])]
    summaries = [str(text) if text else "" for text in batch.get("summary", [])]

    # Tokenize the batch
    model_inputs = tokenizer(contents, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(summaries, padding="max_length", truncation=True, max_length=200)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Apply preprocessing and map the preprocessing to dataset.
train_data = train_data.map(preprocess_data, batched=True)
eval_data = eval_data.map(preprocess_data, batched=True)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 3)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 2)

    # Hyperparameters for training
    training_args = TrainingArguments(
        output_dir="models/t5_book_summarizer",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        fp16=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    print(f"Training with params: {trial.params}")
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")
    return eval_results["eval_loss"]

print("Hyperparameter Tuning Started")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print("Hyperparameters found:")
print(study.best_params)
print(f"Best evaluation loss: {study.best_value}")

model.save_pretrained("models/t5_book_summarizer")
tokenizer.save_pretrained("models/t5_book_summarizer")
print("Model successfully saved")