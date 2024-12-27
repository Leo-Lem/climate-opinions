from src import ClimateChangeOpinions, BlankBert, train

from __params__ import BLANK_MODEL

dataset = ClimateChangeOpinions(model=BLANK_MODEL)
training, validation, testing = dataset.split(.8, .1, .1)

model = BlankBert()

train(model, training, validation)
