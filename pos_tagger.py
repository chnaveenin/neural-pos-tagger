import sys
import os
import torch
from data_process import DataProcess
from LSTM_POS import LSTMTagger
from FFNN_POS import FFNNTagger
from FFNN_runner import data_process_ffnn

if len(sys.argv) != 2:
    print("Usage: python pos_tagger.py <model_flag>")
    sys.exit(1)
    
model_flag = sys.argv[1]

if model_flag == "-f":
    