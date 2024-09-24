# Code-Mixed Humor & Sarcasm Detection


This repo contains the code for the paper: "Improving code-mixed humor and sarcasm detection through multi-tasking
and native sample mixing". We strongly recommend that you run our codes on the same settings with Docker or Anaconda to ensure reproducibility. 

### Requirements

Python 3.6 or higher <br>
Pytorch >= 1.3.0 <br>
transformers  <br>
Pandas, Numpy <br>

### Setup:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/code-mixed-humor-sarcasm-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and prepare the datasets. (Instructions can be found in the data/README.md file)
4. Train and evaluate models using the provided scripts in this directory.

### Dataset

We have used publicly  available datasets. For humor(Hindi) and sarcasm(Hindi), we have translated corresponding English dataset using Google Translate. To run the code for humor and sarcasm, one can change the data path accordingly.

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

```bash
python3 Native_samples_results/{model}.py
```

Select the model from the following:
{0mBERT_Codemix.py, <br> 0XLMR_Codemix.py, <br> 0NB_Codemix.py, <br> 0RF_Codemix.py, <br> 0SVM_Codemix.py, <br> 0mBERT_Combined_(Cod+eng), <br> 0XLMR_Combined_(Cod+eng), <br> 0NB_Combined_(Cod+eng), <br> 0RF_Combined_(Cod+eng), <br> 0SVM_Combined_(Cod+eng), <br> 0mBERT_Combined_(Cod+hind), <br> 0XLMR_Combined_(Cod+hind), <br> 0NB_Combined_(Cod+hind), <br> 0RF_Combined_(Cod+hind), <br> 0SVM_Combined_(Cod+hind), <br> 0mBERT_Combined_(Cod+hind+eng), <br> 0XLMR_Combined_(Cod+hind+eng), <br> 0NB_Combined_(Cod+hind+eng), <br> 0RF_Combined_(Cod+hind+eng), <br> 0SVM_Combined_(Cod+hind+eng)}

### Model Selection

| Category                    | File Names                        |
|-----------------------------|-----------------------------------|
| **Code-mixed only models**  | 0NB_Codemix	                  |
|                             | 0RF_Codemix	                  |
|                             | 0SVM_Codemix                   |
|                             | 0mBERT_Codemix                 |
|                             | 0XLMR_Codemix                   |
| **Combined models (Cod+eng)** | 0NB_Combined_(Cod+eng)         |
|                             | 0RF_Combined_(Cod+eng)          |
|                             | 0SVM_Combined_(Cod+eng)            |
|                             | 0mBERT_Combined_(Cod+eng)            |
|                             | 0XLMR_Combined_(Cod+eng)           |
| **Combined Models (Cod+hind)** | 0NB_Combined_(Cod+hind)       |
|                             | 0RF_Combined_(Cod+hind)         |
|                             | 0SVM_Combined_(Cod+hind)           |
|                             | 0mBERT_Combined_(Cod+hind)           |
|                             | 0XLMR_Combined_(Cod+hind)          |
| **Combined Models (Cod+hind+eng)** | 0NB_Combined_(Cod+hind+eng) |
|                             | 0RF_Combined_(Cod+hind+eng)     |
|                             | 0SVM_Combined_(Cod+hind+eng)       |
|                             | 0mBERT_Combined_(Cod+hind+eng)       |
|                             | 0XLMR_Combined_(Cod+hind+eng)      |



=======
{0mBERT_Codemix.py, <br> 0XLMR_Codemix.py, <br> 0NB_Codemix.py, <br> 0RF_Codemix.py, <br> 0SVM_Codemix.py, <br> 0mBERT_Combined_(Cod+eng), <br> 0XLMR_Combined_(Cod+eng), <br> 0NB_Combined_(Cod+eng), <br> 0RF_Combined_(Cod+eng), <br> 0SVM_Combined_(Cod+eng), <br> 0mBERT_Combined_(Cod+hind), <br> 0XLMR_Combined_(Cod+hind), <br> 0NB_Combined_(Cod+hind), <br> 0RF_Combined_(Cod+hind), <br> 0SVM_Combined_(Cod+hind), <br> 0mBERT_Combined_(Cod+hind+eng), <br> 0XLMR_Combined_(Cod+hind+eng), <br> 0NB_Combined_(Cod+hind+eng), <br> 0RF_Combined_(Cod+hind+eng), <br> 0SVM_Combined_(Cod+hind+eng)}
>>>>>>> parent of 4797c34 (Update README.md)

For Multi-Task model results:

```
python3 Multi-tasking_results/mtl_two_tasks.py

python3 Multi-tasking_results/mtl_three_tasks.py
```
We are using two models (mBERT & XLMR) for multi-task learning. Both model path has been given in the code.
