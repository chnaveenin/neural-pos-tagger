Certainly! Here's the updated README file in Markdown format:

# Part-of-Speech Tagger

This project implements a Part-of-Speech (POS) tagger using Feedforward Neural Network (FFNN) and Long Short-Term Memory (LSTM) models. The tagger assigns grammatical tags to words in a sentence, such as noun, verb, adjective, etc.

## Table of Contents

- [Introduction](#introduction)
- [Files](#files)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Input](#input)
- [Output](#output)
- [Example](#example)
- [Intermediate Files](#intermediate-files)
- [License](#license)

## Introduction

The project consists of several Python scripts for data processing, model implementation, hyperparameter tuning, and inference. The FFNN and LSTM models are trained on annotated corpora to learn the mapping between words and their corresponding POS tags.

## Files

- `data_process.py`: Contains functions for data preprocessing and loading annotated corpora.

- `FFNN_POS.py`: Defines the FFNN Tagger class for training and inference using FFNN model.

- `FFNN_runner.py`: Implements the FFNN Runner class for training and evaluation of FFNN model.

- `LSTM_POS.py`: Defines the LSTM Tagger class for training and inference using LSTM model.

- `LSTM_runner.py`: Implements the LSTM Runner class for training and evaluation of LSTM model.

- `hyperparameter_tuning.py`: Conducts hyperparameter tuning for both FFNN and LSTM models.

- `hyperparameter_tuning_results.txt`: Contains the results of hyperparameter tuning.

- `plot.ipynb`: Jupyter notebook for plotting accuracies from hyperparameter tuning.

- `report.pdf`: A report detailing the project methodology, results, and analysis.

- `pos_tagger.py`: Main script for running the POS tagger interactively.

- `README.md`: This README file.

## Dependencies

The project requires Python 3.x and the following Python libraries:

- `torch`: PyTorch deep learning framework for building and training neural networks.

- `numpy`: Library for numerical computing.

- `pandas`: Library for data manipulation and analysis.

- `matplotlib`: Library for creating static, animated, and interactive visualizations.

- `seaborn`: Library for statistical data visualization.

Install the dependencies using `pip`:

```bash
pip install torch numpy pandas matplotlib seaborn
```

## Usage

To run the POS tagger, execute the following command in the terminal:

```bash
python pos_tagger.py -f -r
```

This command runs both FFNN and LSTM models. After running the script, input a sentence when prompted to get the corresponding tags. Type "quit" to exit the loop and terminate the script.

## Input

The input to the POS tagger is a sentence provided by the user during execution. The tagger processes the sentence and assigns POS tags to each word.

## Output

The output of the POS tagger is a sequence of POS tags corresponding to the words in the input sentence.

## Example

```bash
python pos_tagger.py -f -r
```

Input:
```
Enter a sentence: An apple a day keeps the doctor away
```

Output:
```
["DET", "NOUN", "DET", "NOUN", "VERB", "DET", "NOUN", "ADV"]
```

## Intermediate Files

### data_process.py

- Contains functions for data preprocessing and loading annotated corpora.
- Handles tasks such as tokenization, normalization, and dataset splitting.

### FFNN_POS.py

- Defines the FFNN Tagger class for training and inference using FFNN model.
- Includes methods for building the FFNN model architecture, training, and inference.

### FFNN_runner.py

- Implements the FFNN Runner class for training and evaluation of FFNN model.
- Contains methods for initializing the model, executing the training loop, and testing.

### LSTM_POS.py

- Defines the LSTM Tagger class for training and inference using LSTM model.
- Includes methods for building the LSTM model architecture, training, and inference.

### LSTM_runner.py

- Implements the LSTM Runner class for training and evaluation of LSTM model.
- Contains methods for initializing the model, executing the training loop, and testing.

### hyperparameter_tuning.py

- Conducts hyperparameter tuning for both FFNN and LSTM models.
- Includes functions or classes for searching the hyperparameter space, training models, and evaluating performance.

### hyperparameter_tuning_results.txt

- A text file that stores the results of hyperparameter tuning experiments.
- Contains records of hyperparameter configurations and associated performance metrics.

These intermediate files collectively contribute to the development and optimization of the POS tagger models.