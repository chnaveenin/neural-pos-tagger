import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from FFNN_POS import FFNNTagger
from data_process import DataProcess
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def data_process_ffnn(train_data, P, S, num_classes):
    X = []
    Y = []
    for sentence in train_data:
        for j in range(P, len(sentence) - S):
            X.append(torch.tensor([sentence[i][0] for i in range(j - P, j + S + 1)]))
            Y.append(torch.tensor(sentence[j][1]))
    # apply one hot encoding to Y
    Y = torch.nn.functional.one_hot(torch.tensor(Y), num_classes=num_classes).float()
    X = torch.stack(X)
    return torch.utils.data.TensorDataset(X, Y)

EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_EPOCHS = 2
BATCH_SIZE = 16
P = 2
S = 2


data_process = DataProcess(file_path="./UD_English-Atis/en_atis-ud-train.conllu", flag=True, prev=P, succ=S)
train_data, word_to_ix, tag_to_ix = data_process.get_words_and_tags(data_process.data)

print(tag_to_ix)
print(len(tag_to_ix))

train_data = data_process_ffnn(train_data, P, S, len(tag_to_ix))

train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

VOCAB_SIZE = len(word_to_ix)
TAGSET_SIZE = len(tag_to_ix)

model = FFNNTagger(EMBEDDING_DIM, HIDDEN_DIM, TAGSET_SIZE, P, S, word_to_ix, tag_to_ix)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def get_accuracy(model, validating_data):
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in validating_data:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == np.argmax(y, axis=1)).sum().item()

        print('Accuracy of the network on the %d validation examples: %d %%' %
              (total, 100 * correct / total))
        return (100 * correct / total)

def train_model(model, train_loader, loss_function, optimizer, EPOCHS, validating_data):
    running_loss = 0.0
    prev_accuracy = 0

    for epoch in range(EPOCHS):
        running_loss = 0.0
        print(f"Starting Epoch {epoch}")

        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                running_loss = 0.0

        print(f"Epoch {epoch} completed with loss {running_loss/len(train_loader)}")

        # Calculate and print accuracy on validation data
        prev_accuracy = get_accuracy(model, validating_data)
        print(f"Epoch {epoch} Accuracy: {prev_accuracy}")
        
val_data = data_process.get_data_from_prev("./UD_English-Atis/en_atis-ud-dev.conllu")
val_data = data_process_ffnn(val_data, P, S, len(tag_to_ix))
val_data = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

train_model(model, train_data, criterion, optimizer, NUM_EPOCHS, val_data)

def test_model(model, test_data):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in test_data:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(np.argmax(y, axis=1))
            y_pred.extend(predicted)
    print(classification_report(y_true, y_pred, target_names=tag_to_ix.keys()))
    print(confusion_matrix(y_true, y_pred))
    
test_data = data_process.get_data_from_prev("./UD_English-Atis/en_atis-ud-test.conllu")
test_data = data_process_ffnn(test_data, P, S, len(tag_to_ix))
test_data = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

test_model(model, test_data)