# Implementation Details

All generated files/data is stored in the .out folder.

## Detailed implementation notes.
- Hyperparameters and configuration.
  - supported command line arguments (see also __params__.py): "baseline|blank|sentiment" (model selection), epochs, batch size, sample (refer to [sample mode](#sample-mode))
  - defaults can be configured in __params__.

- Data is preprocessed
  - news label is removed.
  - labels are shifted to 0, 1, 2 (easier to work with non-negative numbers)
  - encoding is done with the BertTokenizer of the model (currently it'll cache the encoded data, which might be problematic for the sentiment model. but then again, maybe it won't, so I didn't fix yet) and the encoded data is stored with attention mask.

- Model is created in Bert class.
  - Blank and baseline are based on huggingface BERT-base-uncased.
  - Sentiment based model not implemented yet.

- Model is trained in BertTrainer class.
  - epochs/batch size can be configured via command line arguments or in code.
  - model parameters, optimizer parameters, epoch and loss are saved after each epoch.
  - best model is selected by (validation) loss and saved after each epoch.
  - model can be loaded at epoch checkpoint if training is interrupted.
  - training progress is displayed with tqdm progress bars.
  - baseline model is not trained.

- Model is evaluated in BertEvaluator class.
  - best model is loaded.
  - evaluation will be run if training is interrupted.
  - evaluation metrics are: accuracy, precision/recall/f1 for each class (negative=0, neutral=1, positive=2).
  - evaluation results are stored in csv with model name and timestamp.

## Sample mode
When developing, you can use the --sample flag to only use a small subset of the data to see if everything works before starting long training processes.