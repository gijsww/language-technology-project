import fasttext
import fasttext.util
import torch
import math
import random
import numpy as np
from model import gan
from sklearn.utils import shuffle

print(torch.version.__version__)

### VARIABLES & ADMINISTRATIVE STUFF ###

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
vocab_size = 2000
languages = {'src': ['de', 'nl'], 'trgt': 'en'}  # Target language to be indicated in last position


### FUNCTIONS ###

def cleaned_vocab(vocab, vocab_size):
    # Remove all punctuation tokens while valid nr of tokens is insufficient yet for having full vocab size
    # TODO & possibly reserve testing vocab

    # Return clean & restricted vocab
    words = vocab.words[:vocab_size]              # Y (labels)
    vects = [vocab[word] for word in words]       # X (input data)

    return vects, words


def add_lang_to_vocab(lang_type, lang_id, vocab_size, vocabs):
    # Get dataset
    if dataset_path == './':
        fasttext.util.download_model(lang_id)  # Download word embedding vector data if not available
    vocab = fasttext.load_model(dataset_path + 'cc.' + lang_id + '.300.bin')  # Load language data

    # Add train data (embedding-vectors) and labels (words) to vocab
    x, y = cleaned_vocab(vocab, vocab_size)
    vocabs[lang_type][lang_id] = {'x': torch.tensor(x), 'y': y}

    return vocabs

def import_dictionary(dictionary_text):
    dictionary = {}
    source = True
    source_word = ''
    target_word = ''
    for character in dictionary_text:
        if source is True:
            if character is '\t':
                source = False
            else:
                source_word = source_word + character
        else:
            if character is '\n':
                source = True
                if source_word in dictionary:
                    dictionary[source_word].append(target_word)
                else:
                    dictionary[source_word] = [target_word]
                source_word = ''
                target_word = ''
            else:
                target_word = target_word + character
    return dictionary


def compute_cosine(vector1, vector2):
    # Computes the cosine simularity between two vectors
    dot_pruduct = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_pruduct/(norm_vector1*norm_vector2)


def get_translation(language, generator, source_embedding, target_vocab):
    # Returns a random translation now, but will return the translation based on
    # the nearest neighbor as given by cosine simularity
    random_word_index = random.randint(0, len(target_vocab['x']-1))
    return target_vocab['x'][random_word_index], target_vocab['y'][random_word_index]


def get_average_cosine(language, generator, source_word_vectors, target_vocab):
    # Computes the average cosine simularity between the source words and their translations
    sum_of_cosines = 0
    for word_vector in source_word_vectors:
        translated_word_vector = get_translation(language, generator, word_vector, target_vocab)[0]
        sum_of_cosines += compute_cosine(word_vector, translated_word_vector)
    return sum_of_cosines/len(source_word_vectors)

def get_translation_accuracy(language, generator, source_words, target_vocab, dictionary):
    # Compute the accuracy of translation over the given set of source words

### MAIN ###

def main():
    nr_src_langs = len(languages)
    real_label, fake_label = 1, 0

    num_minibatches = vocab_size // batch_size

    vocabs = {'src': {}, 'trgt': {}}

    # Get source languages vocab
    for language in languages['src']:
        vocabs = add_lang_to_vocab('src', language, vocab_size, vocabs)

    # Get target language vocab
    language = languages['trgt']
    vocabs = add_lang_to_vocab('trgt', language, vocab_size, vocabs)

    print('Successfully loaded language models.')

    # Get bilingual dictionary for evaluating train loss or at least testing
    dicts = dict()
    #TODO

    # Set up model architecture
    net = gan.GAN(embedding_dim, internal_dim, hidden, languages['src'])

    # Get optimizers; 1 per source language and 1 for target language
    optims_g = {}
    for language in languages['src']:
        params = net.generator.encoders[language].parameters() + net.generator.decoder.parameters()
        optims_g[language] = torch.optim.Adam(params,
                                              lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    optim_d = torch.optim.Adam(net.discriminator.parameters(),
                               lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Train
    train_loss_real_d, train_loss_fake_d, train_loss_fake_g = [], [], []
    eval_loss = [], []
    for epoch in range(epochs):
        print('Epoch ', epoch, '/', epochs)
        loss_real_d_total, loss_fake_d_total, loss_fake_g_total = 0., 0., 0.

        # Shuffle data #
        # Source languages
        for lang in languages['src']:
            vocabs['src'][lang]['x'], vocabs['src'][lang]['y'] = shuffle(vocabs['src'][lang]['x'], vocabs['src'][lang]['y'])
        # Target language
        lang = languages['targt']
        vocabs['trgt'][lang]['x'], vocabs['trgt'][lang]['y'] = shuffle(vocabs['trgt'][lang]['x'], vocabs['trgt'][lang]['y'])

        # Train #
        for batch in range(num_minibatches):
            print('Epoch ', epoch, ', Batch ', batch, '/', num_minibatches)

            # Update discriminator #
            # All-real minibatch
            net.discriminator.zero_grad()
            x = vocabs['trgt'][languages['targt']]['x'][batch * batch_size:(batch + 1) * batch_size].to(device)
            y_true = torch.full((batch_size,), real_label, device=device)
            y_pred = net.discriminator(x)
            loss_real = net.loss(y_pred, y_true, 'dis')
            loss_real.backward()
            loss_real_d_total += loss_real

            # One minibatch per source language
            translations = {}
            loss_fake_batch_avg = 0.
            for language in languages['src']:
                # All-real minibatch
                net.discriminator.zero_grad()
                x = vocabs['src'][language]['x'][batch * batch_size:(batch + 1) * batch_size].to(device)
                x = net.generator(x, language)
                translations[language] = x
                y_true = torch.full((batch_size,), fake_label, device=device)
                y_pred = net.discriminator(x.detach())      # Detach to avoid computing grads for generator
                loss_fake = net.loss(y_pred, y_true, 'dis')
                loss_fake_batch_avg += loss_fake
                loss_fake.backward()    # Compute gradients only for discriminator
            optim_d.step()              # Weight update
            loss_fake_d_total += (loss_fake_batch_avg/nr_src_langs)

            # Update generator #
            # Compute gradients
            loss_fake_batch_avg = 0.
            for language in languages['src']:
                net.generator.encoders[language].zero_grad()
                x = translations[language]
                y_true = torch.full((batch_size,), fake_label, device=device)
                y_pred = net.discriminator(x)
                loss_fake = net.loss(y_pred, y_true, 'gen')
                loss_fake_batch_avg += loss_fake
                loss_fake.backward()
            loss_fake_g_total += (loss_fake_batch_avg / nr_src_langs)

            # TODO: possibly average decoder's gradients

            # Perform weight updates
            for language in languages['src']:
                optims_g[language].step()

        # Document accumulated losses per epoch
        train_loss_real_d.append(loss_real_d_total)
        train_loss_fake_d.append(loss_fake_d_total)
        train_loss_fake_g.append(loss_fake_g_total)

        # TODO: evaluation per epoch?
        # TODO: evaluation could be measured by average cosine between predicted translations

        print('Epoch {}/{}: dr-loss = {}, '
              'df-loss = {}, gf-loss = {}'.format(epoch, epochs,
                                                  train_loss_real_d, train_loss_fake_d, train_loss_fake_g))
    # TODO: Final evaluation

    # TODO: Store model

if __name__ == "__main__":
    # execute only if run as a script
    main()
