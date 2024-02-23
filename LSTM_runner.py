import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from LSTM_POS import LSTMTagger
from data_process import DataProcess
from sklearn.metrics import classification_report, confusion_matrix

class LSTMRunner:
    def __init__(self, train_file, val_file, test_file, embedding_dim=100, hidden_dim=128, lr=0.001, num_epochs=2, batch_size=1, activation='relu', num_layers=1):
        self.data_process_train = DataProcess(train_file)
        
        self.train_data, self.word_to_ix, self.tag_to_ix = self.data_process_train.get_words_and_tags(self.data_process_train.data)
        self.val_data = self.data_process_train.get_data_from_prev(val_file)
        self.test_data = self.data_process_train.get_data_from_prev(test_file)
        
        self.vocab_size = len(self.word_to_ix)
        self.tagset_size = len(self.tag_to_ix)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.activation = activation
        
        self.model = LSTMTagger(self.embedding_dim, self.hidden_dim, self.vocab_size, self.tagset_size, activation, num_layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def get_accuracy(self, model, data_loader):
        with torch.no_grad():
            correct = 0
            total = 0
            for data in data_loader:
                inputs = torch.tensor([ele[0] for ele in data])
                labels = torch.tensor([ele[1] for ele in data])
                inputs = torch.tensor(inputs)
                labels = torch.tensor(labels)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the %d validation examples: %d %%' %
                (total, 100 * correct / total))
            return (100 * correct / total)
    
    def train_model(self):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        val_accuracy = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs = torch.tensor([ele[0] for ele in data])
                labels = torch.tensor([ele[1] for ele in data])
                outputs = self.model(inputs)
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
            val_accuracy = self.get_accuracy(self.model, self.val_data)
            print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")
        return val_accuracy
    
    def test_model(self):
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size)
        test_accuracy = self.get_accuracy(self.model, test_loader)
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        all_labels = []
        all_predicted = []
        for data in test_loader:
            inputs = torch.tensor([ele[0] for ele in data])
            labels = torch.tensor([ele[1] for ele in data])
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_predicted.extend(predicted.numpy())
        print(classification_report(all_labels, all_predicted, target_names=self.tag_to_ix.keys()))
        print(confusion_matrix(all_labels, all_predicted))
        
    def predict(self, sentence):
        sentence = sentence.split()
        for i in range(len(sentence)):
            if sentence[i] not in self.word_to_ix:
                sentence[i] = self.word_to_ix["<UNK>"]
            else:
                sentence[i] = self.word_to_ix[sentence[i]]
        inputs = torch.tensor([sentence[i:i+1] for i in range(len(sentence))])
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted_tags = predicted.numpy().tolist()
        return self.data_process_train.get_tags_from_ix(predicted_tags)