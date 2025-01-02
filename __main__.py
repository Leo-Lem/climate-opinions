from src import ClimateOpinions, Bert, BertTrainer, BertEvaluator

model = Bert.create()

# TODO: tokenization is currently not being reloaded when changing the model
blank_dataset = ClimateOpinions(tokenizer=model.tokenizer)
training, validation, testing = blank_dataset.split(.8, .1, .1)

try:
    train = BertTrainer(model)
    train(training, validation)
except KeyboardInterrupt:
    print("Training interrupted.")

evaluate = BertEvaluator(model)
evaluate(testing)
