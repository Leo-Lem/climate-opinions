from src import ClimateOpinions, Bert, BertTrainer, BertEvaluator, BertPredictor

model = Bert.create()

# TODO: tokenization is currently not being reloaded when changing the model
blank_dataset = ClimateOpinions(tokenizer=model.tokenizer)
training, validation, testing = blank_dataset.split(.8, .1, .1)

try:
    train = BertTrainer(model)
    train(training, validation)
except KeyboardInterrupt:
    print("Training interrupted.")

try:
    evaluate = BertEvaluator(model)
    evaluate(testing)
except KeyboardInterrupt:
    print("Evaluation interrupted.")

predict = BertPredictor(model)
predict.corpus()
