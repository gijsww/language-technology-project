import fasttext
import fasttext.util
import torch
from model import gan

print(torch.version.__version__)

## Variables

# System
dataset_path = '/media/daniel/Elements/FastText_Data/'  # In case dataset is stored somewhere else, e.g. on hard-drive
num_gpus = 0

# Network
embedding_dim = 300
internal_dim = 300
hidden = 300

# Train hyperparameters
vocab_size = 200
languages = ['en', 'de', 'nl']

## Get language data
vocabs = dict()
for language in languages:

    if dataset_path == './':
        fasttext.util.download_model(language)  # Download word embedding vector data
        vocab = fasttext.load_model(dataset_path + 'cc.' + language + '.300.bin')  # Load language data
    else:
        # Assumes datasets are stored somewhere elese already, e.g. on hard-drive
        vocab = fasttext.load_model(dataset_path + 'cc.' + language + '.300.bin')  # Load language data

    # Select vocab
    vocabs[language] = vocab[:vocab_size]

print('Successfully loaded language models.')

# TODO: Get bilingual dictionary (to at least select a few thousand seed words accross languages)



# TODO: Set up model architecture

device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

gan = gan.GAN(embedding_dim, internal_dim, hidden)

# TODO: Train

# TODO: Validate

# TODO: Store model

