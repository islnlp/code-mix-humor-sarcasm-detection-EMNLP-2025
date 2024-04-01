# Code-Mixed Humor & Sarcasm Detection

This repo contains the code for above.

### Requirements

Python 3.6 or higher <br>
Pytorch >= 1.3.0 <br>
transformers  <br>
Pandas, Numpy <br>

### Dataset

We have used publicaly  available datasets. For humor_hindi and sarcasm_hindi we have translated corresponding english data using Google Translate.

### Arguments:

For Multi-tasking_results:

```
--epochs:  number of total epochs to run, default=50

--batch-size: train batchsize, default=32

--lr: learning rate for the model, default=2e-5

--seq_len: sequence lenght of input text, default=128
```

### Training
 For Native_samples_results:

```
python3 Native_samples_results/{model}.py
```
