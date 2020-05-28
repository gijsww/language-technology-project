import fasttext
import fasttext.util
import torch
from model import gan

print(torch.version.__version__)

dataset_path = './'  # In case dataset is stored somewhere else, e.g. on hard-drive

# TODO: Due to size of datasets, probably only one at a time shall be in RAM
if dataset_path == './':
    # English
    fasttext.util.download_model('en')                              # Download English word embedding vectors
    ft_en = fasttext.load_model(dataset_path + 'cc.en.300.bin')     # Load English word embedding vectors

    # German
    fasttext.util.download_model('de')                              # Download German word embedding vectors
    ft_de = fasttext.load_model(dataset_path + 'cc.de.300.bin')     # Load German word embedding vectors

    # Dutch
    fasttext.util.download_model('nl')                              # Download Dutch word embedding vectors
    ft_nl = fasttext.load_model(dataset_path + 'cc.nl.300.bin')     # Load Dutch word embedding vectors
else:
    # Assumes datasets are stored somewhere elese already, e.g. on hard-drive
    ft_en = fasttext.load_model(dataset_path + 'cc.en.300.bin')     # Load English word embedding vectors
    ft_de = fasttext.load_model(dataset_path + 'cc.de.300.bin')     # Load German word embedding vectors
    ft_nl = fasttext.load_model(dataset_path + 'cc.nl.300.bin')     # Load Dutch word embedding vectors

print('Successfully loaded language models.')

# TODO: Get bilingual dictionary (to at least select a few thousand seed words accross languages)

# TODO: Select vocab

# TODO: Set up model architecture
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

embedding_dim = 300
internal_dim = 300
hidden = 300

gan = gan.GAN(embedding_dim, internal_dim, hidden)

# TODO: Train

# TODO: Validate

# TODO: Store model

