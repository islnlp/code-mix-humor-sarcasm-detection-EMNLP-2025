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
Select the model from the following:
{0mBERT_Codemix.py, 0XLMR_Codemix.py, 0NB_Codemix.py, 0RF_Codemix.py, 0SVM_Codemix.py, 0mBERT_Combined_(Cod+eng), 0XLMR_Combined_(Cod+eng), 0NB_Combined_(Cod+eng), 0RF_Combined_(Cod+eng), 0SVM_Combined_(Cod+eng), 0mBERT_Combined_(Cod+hind), 0XLMR_Combined_(Cod+hind), 0NB_Combined_(Cod+hind), 0RF_Combined_(Cod+hind), 0SVM_Combined_(Cod+hind), 0mBERT_Combined_(Cod+hind+eng), 0XLMR_Combined_(Cod+hind+eng), 0NB_Combined_(Cod+hind+eng), 0RF_Combined_(Cod+hind+eng),0SVM_Combined_(Cod+hind+eng)}
