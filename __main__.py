from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src import preprocess, train, evaluate, predict
from __params__ import MODEL

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)

training, validation, testing = preprocess(tokenizer)

trainer = train(model, tokenizer, training, validation)
results = evaluate(trainer, testing)
print(results)

predictions = predict("twitter")
print(predictions.head(50))
