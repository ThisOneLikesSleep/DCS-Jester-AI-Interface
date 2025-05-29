from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import joblib
import spacy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from classification_data import *

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')

sentences = []
labels = []
for key in commands:
    for sentence in commands[key]:
        # lemmatize sentence
        doc = nlp(sentence)
        lemmatized_tokens = []

        for token in doc:
            if token.is_punct or token.is_space:
                continue
            # Preserve case for acronyms
            elif token.text.isupper() and len(token.text) > 1:
                lemmatized_tokens.append(token.text)
            else:
                lemmatized_tokens.append(token.lemma_.lower())

        lemmatized_sentence = " ".join(lemmatized_tokens)
        sentences.append(lemmatized_sentence)

for i in range(59):
    for _ in range(4):
        labels.append(i)

print(sentences)

embeddings = model.encode(sentences)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(embeddings, labels):
    '''
    Trains a model with embedding.
    '''
    normalizer = joblib.load('normalizer.pkl')
    pca = joblib.load('pca.pkl')
    embeddings = normalizer.transform(embeddings)
    embeddings = pca.transform(embeddings)
    embeddings = torch.tensor(embeddings, device='cuda')

    print(embeddings[1])

    label_tensor = torch.tensor(labels, dtype=torch.int64, device='cuda')
    one_hot_tensor = torch.zeros(len(labels), 59, device='cuda')
    one_hot_tensor.scatter_(1, label_tensor.unsqueeze(1), 1)

    print(one_hot_tensor)

    # Parameters
    input_dim = 61
    hidden_dim = 32
    output_dim = 59
    dropout_rate = 0.3
    learning_rate = 0.01
    num_epochs = 100
    batch_size = 16
    patience = 4  # Number of epochs with no improvement after which training will be stopped

    # Create the model, define the loss function and the optimizer
    model = SimpleNN(input_dim, hidden_dim, output_dim, dropout_rate).to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Generate some random data for training (for demonstration purposes)
    # In practice, replace this with your actual dataset

    train_dataset = TensorDataset(embeddings, one_hot_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # Early stopping variables
    best_loss = float('inf')
    epochs_no_improve = 0

    # Training loop with early stopping
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f'Starting epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss:.4f}')

        # Early stopping logic
        if running_loss < best_loss:
            best_loss = running_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'commands.pth')  # Save the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    print('Finished Training')
    return


def visualize_data(embeddings):
    '''
    Visualizes data with t-SNE after PCA is applied. Embeddings with PCA is also saved.
    '''
    normalizer = joblib.load('normalizer.pkl')
    pca = joblib.load('pca.pkl')

    embeddings_norm = normalizer.transform(embeddings)
    embeddings_pca = pca.transform(embeddings_norm)

    tsne = TSNE(n_components=2, random_state=42)
    X_train_tsne = tsne.fit_transform(embeddings_pca)

    # Step 3: Visualize the t-SNE results
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    for i, label in enumerate(labels):
        plt.text(X_train_tsne[i, 0], X_train_tsne[i, 1], str(label), fontsize=9, ha='right')
    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.title('PCA visualization of embeddings with cluster labels')
    plt.colorbar(scatter, label='Cluster Label')
    plt.show()

    return
def calculate_PCA(embeddings):
    '''
    Calculates component number that retains 95% of the variance.
    '''
    # Instantiate PCA and normalizer
    pca = PCA()
    normalizer = Normalizer()

    # Determine transformed features
    embeddings_norm = normalizer.fit_transform(embeddings)
    embeddings_pca = pca.fit_transform(embeddings)

    # Determine explained variance using explained_variance_ratio_ attribute
    exp_var_pca = pca.explained_variance_ratio_

    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    # Find the number of components that explain at least 95% of the variance
    n_components = np.argmax(cum_sum_eigenvalues >= 0.95) + 1
    print(f'Number of components explaining at least 95% of the variance: {n_components}')

    # Transform the data using the selected number of components
    pca = PCA(n_components=n_components)
    X_train_pca_95 = pca.fit_transform(embeddings_norm)
    print(X_train_pca_95.shape)

    # Create the visualization plot
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid', label='Cumulative explained variance')
    plt.axvline(x=n_components - 1, color='r', linestyle='--', label=f'95% variance ({n_components} components)')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    joblib.dump(pca, 'pca.pkl')
    joblib.dump(normalizer, 'normalizer.pkl')

    return

# calculate_PCA(embeddings)
# visualize_data(embeddings)
train_model(embeddings, labels)