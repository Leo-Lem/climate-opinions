from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src import preprocess, train, evaluate
from __params__ import MODEL

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

training, validation, testing = preprocess(tokenizer)

trainer = train(model, tokenizer, training, validation)
results = evaluate(trainer, testing)

print(results)
