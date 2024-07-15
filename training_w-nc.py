from datasets import load_dataset, DatasetDict

# Load the dataset
t_dataset = load_dataset('json', data_files='./new-content-short-body/training/*')
v_dataset = load_dataset('json', data_files='./new-content-short-body/validating/*')

# Combine into a DatasetDict if needed
dataset = DatasetDict({
    'train': t_dataset['train'],
    'validation': v_dataset['train']
})

print(f"Training dataset size: {len(dataset['train'])}")
print(f"Validation dataset size: {len(dataset['validation'])}")

# Tokenization and Data Preprocessing
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

def preprocess_function(examples):
    inputs=examples['match_info']
    targets = examples['article']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=150, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=500,
    logging_dir='./logs',
    logging_steps=100,
)

# Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

# Start Training
trainer.train()

trainer.save_model('./saved-base')
tokenizer.save_pretrained('./saved-base')