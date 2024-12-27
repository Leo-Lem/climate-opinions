from src import ClimateChangeOpinions, finetune_bert

dataset = ClimateChangeOpinions()
train, val, test = dataset.split(.8, .1, .1)
print(train.data.head())
