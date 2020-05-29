import fasttext
import fasttext.util
import torch
import math
from model import gan
from sklearn.utils import shuffle

print(torch.version.__version__)

## Variables

# System
dataset_path = '/media/daniel/Elements/FastText_Data/'  # In case dataset is stored somewhere else, e.g. on hard-drive
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Network
embedding_dim = 300
internal_dim = 300
hidden = 300

# Train hyperparameters
epochs = 10
batch_size = 32
vocab_size = 200
languages = ['de', 'nl', 'en']

## Get language data
vocabs = dict()
for language in languages:

    if dataset_path == './':
        fasttext.util.download_model(language)  # Download word embedding vector data if not available
    vocab = fasttext.load_model(dataset_path + 'cc.' + language + '.300.bin')  # Load language data

    # Select vocab -- TODO: possibly filter out non-textual stuff
    words = vocab.words[:vocab_size]                # Y (labels)
    subset_x = [vocab[word] for word in words]      # X (input data)
    vocabs[language] = {'x': subset_x, 'y': words}

#print(vocabs)
print('Successfully loaded language models.')

# TODO: Get bilingual dictionary (to at least select a few thousand seed words across languages)
dicts = dict()
#...

# Set up model architecture
gan = gan.GAN(embedding_dim, internal_dim, hidden)
optimizer = torch.optim.Adam(gan.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Train
train_loss, eval_loss = [], []
src_1_x, src_1_y = vocabs[languages[0]]['x'], vocabs[languages[0]]['y']
src_2_x, src_2_y = vocabs[languages[1]]['x'], vocabs[languages[1]]['y']
targt_x, targt_y = vocabs[languages[2]]['x'], vocabs[languages[2]]['y']

for epoch in range(epochs):
    # Shuffle data
    src_1_x, src_1_y = shuffle(src_1_x, src_1_y)
    src_2_x, src_2_y = shuffle(src_2_x, src_2_y)
    targt_x, targt_y = shuffle(targt_x, targt_y)

    # Iterate through all mini-batches

    for batch_idx in range((len(src_1_x) + int(math.ceil(batch_size/3)) - 1) // int(math.ceil(batch_size/3))):
        # TODO: Double-check
        # Construct mini-batch: [de_embed1, de_embed2,... nl_embed1, nl_embed2,...]
        # Hence, minibatch contains 2 partitions: 1. German embedding data, 2. Dutch embedding data
        # For labels, a third partition is added which contains the data to confuse the discriminator. Hence,
        # roughly 1/3 for each partition is used, hence the division of batch sizes.

        inputs = torch.tensor(src_1_x[batch_idx * int(math.ceil(batch_size/3)): (batch_idx + 1) * int(math.ceil(batch_size/3))] +
                              src_2_x[batch_idx * int(math.ceil(batch_size/3)): (batch_idx + 1) * int(math.ceil(batch_size/3))])

        inputs_fake = torch.tensor(targt_x[batch_idx * int(math.ceil(batch_size/3)):
                                           (batch_idx + 1) * int(math.ceil(batch_size/3))])

        # First X elements are supposed to contain 0 for GAN generated vecs and 1 for vectors native to target space
        labels = torch.cat((torch.zeros(inputs.shape[0]), torch.ones(inputs_fake.shape[0])), 0)

        # Reset parameter gradients
        optimizer.zero_grad()

        # Forward, Backward, Optimize
        outputs = gan(inputs, inputs_fake)               # Outputs: per batch element: [p(TP), p(FP)]
        loss = gan.loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0



# TODO: Validate

# TODO: Store model

