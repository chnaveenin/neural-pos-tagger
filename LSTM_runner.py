import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from LSTM_POS import LSTMTagger
from data_process import DataProcess
from sklearn.metrics import classification_report, confusion_matrix

data_process = DataProcess("./UD_English-Atis/en_atis-ud-train.conllu")
train_data, word_to_ix, tag_to_ix = data_process.get_words_and_tags(data_process.data)

EMBEDDING_DIM = 100
HIDDEN_DIM = 128
VOCAB_SIZE = len(word_to_ix)
TAGSET_SIZE = len(tag_to_ix)
NUM_EPOCHS = 2

# Create the model
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAGSET_SIZE, 1)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

def get_accuracy(model, validating_data):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in validating_data:
            inputs = []
            labels = []
            for ele in data:
                inputs.append(ele[0])
                labels.append(ele[1])
            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)
            outputs = model(inputs)
            # print outputs from valid_tag_to_ix to get the actual tags
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %d validation examples: %d %%' %
              (total, 100 * correct / total))
        return (100 * correct / total)

def train_model(model, training_data, loss_function, optimizer, EPOCHS, BATCH_SIZE, validating_data, tag_to_ix):
    running_loss = 0.0
    prev_accuracy = 0

    # Create DataLoader for batch training
    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        print(f"Starting Epoch {epoch}")

        # Set model to training mode
        model.train()

        for i, data_ in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = []
            labels = []
            for ele in data_:
                inputs.append(ele[0])
                labels.append(ele[1])
            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch} completed with loss {running_loss/len(train_loader)}")

        # Calculate and print accuracy on validation data
        prev_accuracy = get_accuracy(model, validating_data)
        print(f"Epoch {epoch} Accuracy: {prev_accuracy}")
        
val_data = data_process.get_data_from_prev("./UD_English-Atis/en_atis-ud-dev.conllu")
        
train_model(model, train_data, criterion, optimizer, NUM_EPOCHS, 1, val_data, tag_to_ix)

def test_model(model, testing_data, train_tag_to_ix):
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        for data in testing_data:
            inputs = []
            labels = []
            for ele in data:
                inputs.append(ele[0])
                labels.append(ele[1])
            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)
            outputs = model((inputs))
            outputs = model(inputs)
            # print outputs from valid_tag_to_ix to get the actual tags
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            labels = (labels.numpy())
            predicted = (predicted.numpy())
            for i in labels:
                all_labels.append(i)
            for i in predicted:
                all_predicted.append(i)
        print('Accuracy of the network on the %d test examples: %d %%' %
              (total, 100 * correct / total))
        # get precision, recall and f1 score
        print(classification_report(all_labels, all_predicted,
              target_names=list(train_tag_to_ix.keys()).remove("<UNK>")))
        print(confusion_matrix(all_labels, all_predicted))

test_data = data_process.get_data_from_prev("./UD_English-Atis/en_atis-ud-test.conllu")
test_model(model, test_data, tag_to_ix)