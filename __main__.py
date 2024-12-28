from src import ClimateOpinions, BlankBert, BertTrainer, BertEvaluator

blank_model = BlankBert()

blank_dataset = ClimateOpinions(tokenizer=blank_model.tokenizer)
training, validation, testing = blank_dataset.split(.8, .1, .1)

try:
    blank_trainer = BertTrainer(blank_model)
    blank_trainer(training, validation)
except KeyboardInterrupt:
    print("Training interrupted.")

blank_evaluator = BertEvaluator(blank_model)
accuracy, precision, recall, f1 = blank_evaluator(testing)
print(
    f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
