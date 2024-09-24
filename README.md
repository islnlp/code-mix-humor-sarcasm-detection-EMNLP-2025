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

Select the model from the following:

<form> <label for="model">Choose a model:</label> <select id="model" name="model"> <option value="0mBERT_Codemix.py">0mBERT_Codemix.py</option> <option value="0XLMR_Codemix.py">0XLMR_Codemix.py</option> <option value="0NB_Codemix.py">0NB_Codemix.py</option> <option value="0RF_Codemix.py">0RF_Codemix.py</option> <option value="0SVM_Codemix.py">0SVM_Codemix.py</option> <option value="0mBERT_Combined_Cod+eng.py">0mBERT_Combined_(Cod+eng).py</option> <option value="0XLMR_Combined_Cod+eng.py">0XLMR_Combined_(Cod+eng).py</option> <option value="0NB_Combined_Cod+eng.py">0NB_Combined_(Cod+eng).py</option> <option value="0RF_Combined_Cod+eng.py">0RF_Combined_(Cod+eng).py</option> <option value="0SVM_Combined_Cod+eng.py">0SVM_Combined_(Cod+eng).py</option> <option value="0mBERT_Combined_Cod+hind.py">0mBERT_Combined_(Cod+hind).py</option> <option value="0XLMR_Combined_Cod+hind.py">0XLMR_Combined_(Cod+hind).py</option> <option value="0NB_Combined_Cod+hind.py">0NB_Combined_(Cod+hind).py</option> <option value="0RF_Combined_Cod+hind.py">0RF_Combined_(Cod+hind).py</option> <option value="0SVM_Combined_Cod+hind.py">0SVM_Combined_(Cod+hind).py</option> <option value="0mBERT_Combined_Cod+hind+eng.py">0mBERT_Combined_(Cod+hind+eng).py</option> <option value="0XLMR_Combined_Cod+hind+eng.py">0XLMR_Combined_(Cod+hind+eng).py</option> <option value="0NB_Combined_Cod+hind+eng.py">0NB_Combined_(Cod+hind+eng).py</option> <option value="0RF_Combined_Cod+hind+eng.py">0RF_Combined_(Cod+hind+eng).py</option> <option value="0SVM_Combined_Cod+hind+eng.py">0SVM_Combined_(Cod+hind+eng).py</option> </select> </form> 
```

For Multi-Task model results:

```
python3 Multi-tasking_results/mtl_two_tasks.py

python3 Multi-tasking_results/mtl_three_tasks.py
```
We are using two models (mBERT & XLMR) for multi-task learning. Both model path has been given in the code.
