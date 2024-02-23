import sys
from FFNN_runner import FFNNRunner
from LSTM_runner import LSTMRunner

if len(sys.argv) != 2:
    print("Usage: python pos_tagger.py <model_flag>")
    sys.exit(1)
    
model_flag = sys.argv[1]

EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_EPOCHS = 1
BATCH_SIZE = 1
NUM_LAYERS = 1

file_paths = "UD_English-Atis/en_atis-ud-"

trainer = None

if model_flag == "-f":
    trainer = FFNNRunner(
        train_file=file_paths + "train.conllu",
        val_file=file_paths + "dev.conllu",
        test_file=file_paths + "test.conllu",
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        pre=1,
        suc=1,
        lr=0.001
    )

    trainer.train_model()
    trainer.test_model()
        
elif model_flag == "-r":
    trainer = LSTMRunner(
        train_file=file_paths + "train.conllu",
        val_file=file_paths + "dev.conllu",
        test_file=file_paths + "test.conllu",
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        num_layers=NUM_LAYERS
    )
    trainer.train_model()
    trainer.test_model()
    
else:
    print("Usage: model_flag must be -f or -r.")
    sys.exit(1)
    
while True:
    sentence = input("Enter a sentence: ")
    if sentence == "exit":
        break
    print(trainer.predict(sentence))