# Code-Mixed Humor & Sarcasm Detection

This repo contains the code for above.

### Requirements

Python 3.6 or higher <br>
Pytorch >= 1.3.0 <br>
transformers  <br>
Pandas, Numpy <br>

### Dataset

We have used publicly  available datasets. For humor_hindi and sarcasm_hindi we have translated corresponding english data using Google Translate. To run the code for humor and sarcasm, one can change the data path accordingly.

### Arguments:

For Multi-Task model results:

```
--epochs:  number of total epochs to run, default=50

--batch-size: train batchsize, default=32

--lr: learning rate for the model, default=2e-5

--seq_len: sequence lenght of input text, default=128
```

### Training
 For results on native samples mixing:

```
python3 Native_samples_results/{model}.py
```
Select the model from the following:

{0mBERT_Codemix.py, 0XLMR_Codemix.py, <br> 0NB_Codemix.py, <br> 0RF_Codemix.py, <br> 0SVM_Codemix.py, <br> 0mBERT_Combined_(Cod+eng), <br> 0XLMR_Combined_(Cod+eng), <br> 0NB_Combined_(Cod+eng), <br> 0RF_Combined_(Cod+eng), <br> 0SVM_Combined_(Cod+eng), <br> 0mBERT_Combined_(Cod+hind), <br> 0XLMR_Combined_(Cod+hind), <br> 0NB_Combined_(Cod+hind), <br> 0RF_Combined_(Cod+hind), <br> 0SVM_Combined_(Cod+hind), <br> 0mBERT_Combined_(Cod+hind+eng), <br> 0XLMR_Combined_(Cod+hind+eng), <br> 0NB_Combined_(Cod+hind+eng), <br> 0RF_Combined_(Cod+hind+eng), <br> 0SVM_Combined_(Cod+hind+eng)}

For Multi-Task model results:

```
python3 Multi-tasking_results/mtl_two_tasks.py

python3 Multi-tasking_results/mtl_three_tasks.py
```
We are using two models (mBERT & XLMR) for multi-task learning. Both model path has been given in the code, we can use them accordingly.
