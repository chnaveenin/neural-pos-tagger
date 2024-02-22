import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNN_POS_Tagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, prev_context_size, succ_context_size):
        super(FFNN_POS_Tagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * (prev_context_size + 1 + succ_context_size), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.prev_context_size = prev_context_size
        self.succ_context_size = succ_context_size

    def forward(self, inputs):
        prev_inputs = inputs[:, :self.prev_context_size]
        current_input = inputs[:, self.prev_context_size]
        succ_inputs = inputs[:, self.prev_context_size + 1:]

        # Embed previous, current, and successive tokens separately
        prev_embedded = self.embedding(prev_inputs)
        current_embedded = self.embedding(current_input)
        succ_embedded = self.embedding(succ_inputs)

        # Flatten embeddings
        prev_embedded = prev_embedded.view(prev_embedded.size(0), -1)
        succ_embedded = succ_embedded.view(succ_embedded.size(0), -1)

        # Concatenate embeddings
        combined_embedded = torch.cat((prev_embedded, current_embedded, succ_embedded), dim=1)

        # Pass through the fully connected layers
        out_fc1 = F.relu(self.fc1(combined_embedded))
        out_fc2 = self.fc2(out_fc1)
        return F.log_softmax(out_fc2, dim=1)