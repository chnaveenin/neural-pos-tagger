import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from FFNN_POS import FFNNTagger
from data_process import DataProcess, data_process_ffnn
from sklearn.metrics import classification_report, confusion_matrix

class FFNNRunner:
    def __init__(
        self, 
        train_file='', 
        val_file='', 
        test_file='', 
        embedding_dim='', 
        hidden_dim='', 
        num_epochs=2, 
        batch_size=1,
        pre=1, 
        suc=1,
        lr=0.001
    ):
        self.data_process_train = DataProcess(train_file)
        
        self.train_data, self.word_to_ix, self.tag_to_ix = self.data_process_train.get_words_and_tags(self.data_process_train.data)
        self.val_data = self.data_process_train.get_data_from_prev(val_file)
        self.test_data = self.data_process_train.get_data_from_prev(test_file)
        
        self.train_data = data_process_ffnn(self.train_data, pre, suc, len(self.tag_to_ix))
        self.val_data = data_process_ffnn(self.val_data, pre, suc, len(self.tag_to_ix))
        self.test_data = data_process_ffnn(self.test_data, pre, suc, len(self.tag_to_ix))
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Create the FFNN model
        self.model = FFNNTagger(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=len(self.tag_to_ix),
            pre=pre,
            suc=suc,
            vocab_size=len(self.word_to_ix)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def get_accuracy(self, model, data_loader):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in data_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == np.argmax(labels, axis=1)).sum().item()
            accuracy = 100 * correct / total
        return accuracy
    
    def train_model(self):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
            val_accuracy = self.get_accuracy(self.model, DataLoader(self.val_data, batch_size=self.batch_size))
            print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")
    
    def test_model(self):
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size)
        test_accuracy = self.get_accuracy(self.model, test_loader)
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        all_labels = []
        all_predicted = []
        for inputs, labels in test_loader:
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(np.argmax(labels, axis=1))
            all_predicted.extend(predicted.numpy())
        print(classification_report(all_labels, all_predicted, target_names=self.tag_to_ix.keys()))
        print(confusion_matrix(all_labels, all_predicted))

# Define model parameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_EPOCHS = 2
BATCH_SIZE = 1

file_paths = "UD_English-Atis/en_atis-ud-"

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
