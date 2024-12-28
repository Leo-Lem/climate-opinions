from src import ClimateOpinions, BlankBert, BertTrainer

blank_model = BlankBert()

blank_dataset = ClimateOpinions(tokenizer=blank_model.tokenizer)
training, validation, testing = blank_dataset.split(.8, .1, .1)

blank_trainer = BertTrainer(blank_model)
blank_trainer(training, validation)
