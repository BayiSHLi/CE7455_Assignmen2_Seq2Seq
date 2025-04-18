from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import argparse
import torch
from torch import optim
from sklearn.model_selection import train_test_split
import time
import math

from models import *

from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
import numpy as np

import wandb


teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 15

eng_prefixes = (
    "i am", "i m",
    "he is", "he s",
    "she is", "she s",
    "you are", "you re",
    "we are", "we re",
    "they are", "they re"
)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device=device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device=device)
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate(encoder, decoder, sentence, device, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

# Modified training function for bi-LSTM
def train_biLSTM(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    # Encoder forward pass
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # Decoder initial input
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # Prepare decoder hidden state
    if isinstance(encoder_hidden, tuple):  # LSTM case
        # Sum bidirectional states to get single direction
        decoder_hidden = (encoder_hidden[0][0:1] + encoder_hidden[0][1:2],
                          encoder_hidden[1][0:1] + encoder_hidden[1][1:2])
    else:  # GRU case
        decoder_hidden = encoder_hidden[0:1]  # Take first layer

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

# Modified evaluation function for bi-LSTM
def evaluate_biLSTM(encoder, decoder, sentence, device, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        # Prepare decoder hidden state
        if isinstance(encoder_hidden, tuple):  # LSTM case
            decoder_hidden = (encoder_hidden[0][0:1], encoder_hidden[1][0:1])  # Take first layer, forward direction
        else:  # GRU case
            decoder_hidden = encoder_hidden[0:1]  # Take first layer

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

# Modified train function for Attention
def train_Attention(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Store all encoder outputs
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # Encoder forward pass
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # Decoder initial setup
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate_Attention(encoder, decoder, sentence, device, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

# Modified training function for Transformer
def train_transformer(input_tensor, target_tensor, encoder, decoder,
                      encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # Input needs to be (seq_len, 1)
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    for ei in range(input_length):
        encoder_outputs[ei] = encoder_output[0, 0]

    # Decoder initial setup
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # Use the sentence representation as initial hidden state
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# Modified evaluation function for Transformer
def evaluate_Transformer(encoder, decoder, sentence, device, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]

        # Transformer processes entire sequence at once
        encoder_output, encoder_hidden = encoder(input_tensor, None)

        # Create encoder outputs for attention (if used)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(min(input_length, max_length)):
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, None)

            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def trainIters(task, encoder, decoder, epochs, device, print_every=1000, plot_every=100, learning_rate=0.01, ):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = torch.nn.NLLLoss()

    iter = 1
    n_iters = len(train_pairs) * epochs

    for epoch in range(epochs):
        print("Epoch: %d/%d" % (epoch, epochs))
        for training_pair in train_pairs:
            training_pair = tensorsFromPair(training_pair, device=device)
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            if task in ['GRU','LSTM']:
                loss = train(input_tensor, target_tensor, encoder,
                            decoder, encoder_optimizer, decoder_optimizer, criterion)

            elif task == 'bi-LSTM':
                loss = train_biLSTM(input_tensor, target_tensor, encoder,
                             decoder, encoder_optimizer, decoder_optimizer, criterion)

            elif task == 'Attention':
                loss = train_Attention(input_tensor, target_tensor, encoder,
                             decoder, encoder_optimizer, decoder_optimizer, criterion)

            elif task == 'Transformer':
                loss = train_transformer(input_tensor, target_tensor, encoder,
                                       decoder, encoder_optimizer, decoder_optimizer, criterion)

            else:
                raise ValueError("Invalid task. Please choose from: GRU, LSTM, bi-LSTM, Attention, Transformer")

            print_loss_total += loss
            plot_loss_total += loss

            if iter % 100 == 0:
                wandb.log({f"{task}: train loss": plot_loss_total/iter})

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            iter +=1


def evaluateRandomly(task, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])

        if task in ['GRU','LSTM']:
            output_words = evaluate(encoder, decoder, pair[0], device)

        elif task == 'Attention':
            output_words = evaluate_Attention(encoder, decoder, pair[0], device)

        elif task == 'bi-LSTM':
            output_words = evaluate_biLSTM(encoder, decoder, pair[0], device)

        elif task == 'Transformer':
            output_words = evaluate_Transformer(encoder, decoder, pair[0], device)

        else:
            raise ValueError("Invalid task. Please choose from: GRU, LSTM, bi-LSTM, Attention, Transformer")

        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def inference(task, encoder, decoder, testing_pairs):
    input = []
    gt = []
    predict = []

    from tqdm import tqdm
    for i in tqdm(range(len(testing_pairs))):
        pair = testing_pairs[i]
        if task in ['GRU','LSTM']:
            output_words = evaluate(encoder, decoder, pair[0], device)
        elif task == 'Attention':
            output_words = evaluate_Attention(encoder, decoder, pair[0], device)
        elif task == 'bi-LSTM':
            output_words = evaluate_biLSTM(encoder, decoder, pair[0], device)

        elif task == 'Transformer':
            output_words = evaluate_Transformer(encoder, decoder, pair[0], device)

        else:
            raise ValueError("Invalid task. Please choose from: GRU, LSTM, bi-LSTM, Attention, Transformer")
        output_sentence = ' '.join(output_words)

        input.append(pair[0])
        gt.append(pair[1])
        predict.append(output_sentence)

    return input,gt,predict


def eval(gt, predict):
  rouge = ROUGEScore()
  metric_score = rouge(predict, gt)
  print("=== Evaluation score - Rouge score ===")
  print("Rouge1 fmeasure:\t",metric_score["rouge1_fmeasure"].item())
  print("Rouge1 precision:\t",metric_score["rouge1_precision"].item())
  print("Rouge1 recall:  \t",metric_score["rouge1_recall"].item())
  print("Rouge2 fmeasure:\t",metric_score["rouge2_fmeasure"].item())
  print("Rouge2 precision:\t",metric_score["rouge2_precision"].item())
  print("Rouge2 recall:  \t",metric_score["rouge2_recall"].item())
  print("=====================================")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seq2Seq with GRU')
    parser.add_argument('--task', type=str, help='Task to run: GRU, LSTM, bi-LSTM, Attention, Transformer')
    parser.add_argument('--n_epochs', type=int, default=20, help='epochs to train for')
    parser.add_argument('--gpu', type=str, default=0, help='device to use')
    parser.add_argument('--eval', action='store_true', help='eval mode')
    parser.add_argument('--encoder_ckpt', type=str, default='', help='the path to the encoder checkpoint to load')
    parser.add_argument('--decoder_ckpt', type=str, default='', help='the path to the decoder checkpoint to load')

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Prepare data
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    X = [i[0] for i in pairs]
    y = [i[1] for i in pairs]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    train_pairs = list(zip(X_train, y_train))
    test_pairs = list(zip(X_test, y_test))

    hidden_size = 256

    if args.task == 'GRU':
        # Task 2 Example GRU
        encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
        decoder1 = Decoder(hidden_size, output_lang.n_words, device).to(device)

    elif args.task == 'LSTM':
        # Task 3 LSTM
        encoder1 = EncoderLSTM(input_lang.n_words, hidden_size, device).to(device)
        decoder1 = DecoderLSTM(hidden_size, output_lang.n_words, device).to(device)

    elif args.task == 'bi-LSTM':
        # Task 4 bi-LSTM
        encoder1 = EncoderbiLSTM(input_lang.n_words, hidden_size, device).to(device)
        decoder1 = DecoderbiLSTM(hidden_size, output_lang.n_words, device).to(device)

    elif args.task == 'Attention':
        # Task 5 Attention
        encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
        decoder1 = AttnDecoderGRU(hidden_size, output_lang.n_words, device, ).to(device)

    elif args.task == 'Transformer':
        encoder1 = TransformerEncoder(input_lang.n_words, hidden_size, device).to(device)
        decoder1 = TransformerDecoder(hidden_size, output_lang.n_words, device, ).to(device)
    else:
        raise ValueError("Invalid task. Please choose from: GRU, LSTM, bi-LSTM, Attention, Transformer")


    if not args.eval:
        wandb.init(project='Assignment 2',
                   name=args.task,
                   reinit=True)

        trainIters(args.task, encoder1, decoder1, args.n_epochs, print_every=5000, device=device)
        # Save the model
        torch.save(encoder1.state_dict(), f'./ckpt/encoder-{args.task}-e{args.n_epochs}.pt')
        torch.save(decoder1.state_dict(), f'./ckpt/decoder-{args.task}-e{args.n_epochs}.pt')

        evaluateRandomly(args.task, encoder1, decoder1)

        input, gt, predict = inference(args.task, encoder1, decoder1, test_pairs)
        eval(gt, predict)


    else:
        encoder1.load_state_dict(torch.load(args.ckpt))
        decoder1.load_state_dict(torch.load(args.ckpt))

        print("=== Evaluation ===")
        evaluateRandomly(args.task, encoder1, decoder1)
        input, gt, predict = inference(args.task, encoder1, decoder1, test_pairs)
        eval(gt, predict)









