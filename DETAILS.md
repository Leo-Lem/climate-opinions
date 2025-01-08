# Details

## Folder structure
- [\_\_main\_\_.py](__main__.py) is the entry point for the program.
- [\_\_params\_\_.py](__params__.py) contains the hyperparameters, configuration and command line arguments.
- [src](src) is where the source code is stored.
  - [data.py](src/data.py) contains the data loading and preprocessing functions.
  - [model.py](src/model.py) contains the model classes.
  - [train.py](src/train.py) contains the training loop.
  - [eval.py](src/eval.py) contains the evaluation functions.
  - [pred.py](src/pred.py) contains the prediction functions.
- [res](res) is where the resources are stored.
- [.out](.out) is where the generated files are stored.
- [.devcontainer](.devcontainer) stores a VSCode devcontainer configuration.
- [requirements.txt](requirements.txt) lists the required packages with versions.

## Setup

```sh
pip install -r requirements.txt
```


## Implementation
### Hyperparameters and configuration
```sh
python social-media-sentiment-analysis <model> \
  --epochs <epochs> \
  --batch_size <batch_size> \
  --sample
  --results=<results-directory>
```

Defaults can be configured in [\_\_params\_\_.py](__params__.py).
- 'baseline|blank|sentiment' (model selection)
- epochs
- batch size
- seed for random number generation
- results directory to store the best model, evaluation results and predictions

#### Sample mode

When developing, you can use the --sample flag to only use a small subset of the data to see if everything works before starting long training processes.

### Data Preprocessing
Data is preprocessed in the ClimateOpinions class.
- news label is removed.
- labels are shifted to 0, 1, 2 (easier to work with non-negative numbers)
- encoding is done with the BertTokenizer of the model (currently it'll cache the encoded data, which might be problematic for the sentiment model. but then again, maybe it won't, so I didn't fix yet) and the encoded data is stored with attention mask.

### Model
Model is created in Bert class.
- Blank and baseline are based on huggingface BERT-base-uncased.
- Sentiment based model not implemented yet.

### Training
Model is trained in BertTrainer class.
- epochs/batch size can be configured via command line arguments or in code.
- model parameters, optimizer parameters, epoch and loss are saved after each epoch.
- best model is selected by (validation) loss and saved after each epoch.
- model can be loaded at epoch checkpoint if training is interrupted.
- training progress is displayed with tqdm progress bars.
- baseline model is not trained.

### Evaluation
Model is evaluated in BertEvaluator class.
- best model is loaded.
- evaluation will be run if training is interrupted.
- evaluation metrics are: accuracy, precision/recall/f1 for each class (negative=0, neutral=1, positive=2).
- evaluation results are stored in csv with model name and timestamp.

### Prediction
Model is used for prediction in BertPredictor class.
- best model is loaded.
- prediction is done on test data.
- prediction results are stored in csv for selected model.

# [README](README.md)