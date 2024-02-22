import torch.nn as nn
class FFNNTagger(nn.Module):
    def __init__(self, embedding_dim=100, hidden_dim=50, output_dim=1, pre=1, suc=1, vocab_size=0):
        super(FFNNTagger, self).__init__()
        self.p = pre
        self.s = suc
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.fc1 = nn.Linear((pre+suc+1)*embedding_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x