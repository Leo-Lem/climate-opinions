from src import ClimateOpinions, BlankBert, BertTrainer

from __params__ import BLANK_MODEL

dataset = ClimateOpinions(model=BLANK_MODEL)
training, validation, testing = dataset.split(.8, .1, .1)

blank_model = BlankBert()
blank_trainer = BertTrainer(blank_model)
blank_trainer(training, validation)
