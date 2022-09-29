import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as utils

import random

random.seed(0)

#if running on colab, change this variable to true
colab = 0

if not colab:
    import wget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS
batch_size = 32
bptt_len = 64 #change this for testing ex 1.3.6
print_step = 10


class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>'):
        self.id_to_string = {}
        self.string_to_id = {}

        # add the default pad token
        # pad token has id = 0
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0

        # add the default unknown token
        # unk token has id =1
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1

    def __len__(self):
        return len(self.id_to_string)

    # if a new word is captured, id = len(ids) (last element in the voc)
    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    # if not present and extend_vocab = true, then add new word e return new id, otherwise return unknown id
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


class LongTextData:

    def __init__(self, file_path, vocab=None, extend_vocab=True, device='cuda'):
        self.data, self.vocab = self.text_to_data(file_path, vocab, extend_vocab, device)

    def __len__(self):
        return len(self.data)

    def text_to_data(self, text_file, vocab, extend_vocab, device):
        """Read a raw text file and create its tensor and the vocab.

        Args:
          text_file: a path to a raw text file.
          vocab: a Vocab object
          extend_vocab: bool, if True extend the vocab
          device: device

        Returns:
          Tensor representing the input text, vocab file

        """
        assert os.path.exists(text_file)
        if vocab is None:
            vocab = Vocabulary()

        data_list = []

        # Construct data

        # full text will contain all the ids of the corresponding word (to retreive the word associated to each id, go to vocabolary)
        full_text = []
        print(f"Reading text file from: {text_file}")
        with open(text_file, 'r') as text:
            # read each line of the text
            for line in text:
                tokens = list(line)
                for token in tokens:
                    # get index will extend the vocab if the input
                    # token is not yet part of the text.
                    full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))

        # convert to tensor
        data = torch.tensor(full_text, device=device, dtype=torch.int64)
        print("Done.")

        return data, vocab

# Since there is no need for schuffling the data, we just have to split
# the text data according to the batch size and bptt length.
# The input to be fed to the model will be batch[:-1]
# The target to be used for the loss will be batch[1:]
class ChunkedTextData:

    def __init__(self, data, bsz, bptt_len, pad_id):
        self.batches = self.create_batch(data, bsz, bptt_len, pad_id)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def create_batch(self, input_data, bsz, bptt_len, pad_id):
        """Create batches from a TextData object .

        Args:
          input_data: a TextData object.
          bsz: int, batch size
          bptt_len: int, bptt length
          pad_id: int, ID of the padding token

        Returns:
          List of tensors representing batches

        """
        batches = []  # each element in `batches` is (len, B) tensor
        text_len = len(input_data)

        # La lunghezza di ogni batch deve essere lunghezza dell'input data // batch_size + 1
        segment_len = text_len // bsz + 1

        # Question: Explain the next two lines!

        # This creates a tensor which lenght is segment_len * bsz filled by pad_id.
        # We are creating a unique 'pad_id' tensor

        padded = input_data.data.new_full((segment_len * bsz,), pad_id)

        # We filled the padded tensor with data that we know
        padded[:text_len] = input_data.data


        # we reshape the tensor in order to create n_batch with bsz lenght.
        padded = padded.view(bsz, segment_len).t()

        num_batches = segment_len // bptt_len + 1

        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.
            if i == 0:
                # Append a dummy start symbol using pad token
                batch = padded[i * bptt_len:(i + 1) * bptt_len]
                batch = torch.cat(
                    [padded.new_full((1, bsz), pad_id),
                     padded[i * bptt_len:(i + 1) * bptt_len]], dim=0)
                batches.append(batch)
            else:
                batches.append(padded[i * bptt_len - 1:(i + 1) * bptt_len])

        return batches


#change the path if using colab
if (colab):
    text_path = "/content/49010-0.txt"
else:
    text_path = "./49010-0.txt"


#test with this program itself
#text_path = './main.py'

if (colab):
    '''If using colab, uncomment the following line to download the Book'''
    print('\n \t --- Downloading the book --- \t \n', 'REMEMBER TO UNCOMMENT THE DOWNLOADING LINE \n')
    #!wget http://www.gutenberg.org/files/49010/49010-0.txt
else:
    # check if the file has already been downloaded
    if not os.path.exists(text_path):
        url = 'http://www.gutenberg.org/files/49010/49010-0.txt'
        filename = wget.download(url)

filename = os.open(text_path, os.O_RDWR)

#convert text in ids and corresponding vocabulary
my_data = LongTextData(text_path, device=device)

'''Es 1.1.1 Get some features of the text '''


print('\n \t--- DATASET INFO --- \t \n')
#vocabulary size
print("Vocabulary Size: ", len(my_data.vocab.id_to_string.items()))
#print(my_data.vocab.id_to_string)

#characters present in the file
print("Different Characters in the text: ", len(my_data.data))

#get other infos of the text
with open(filename, 'r') as text:
    lines = text.readlines()
    print("Total number of lines: ", len(lines))
    list_len = [len(i) for i in lines]
    print("Maximum line-lenght: ", max(list_len))
    print("Minimum line-lenght: ", min(list_len))
    emptyLines = np.where(np.array(list_len) == min(list_len))
    print("Number of empty lines: ", len(emptyLines[0]))
    print("\n")


batches = ChunkedTextData(my_data, batch_size, bptt_len, pad_id=0)
'''Ex 1.3.1: implement the LSTM '''
class Net(nn.Module):

    def __init__(self, vocabulary_dim, output_dim, embedding_dim=64, hidden =2048, layers=1):
        super().__init__()

        self.num_layers = layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden
        self.vocab_dim = vocabulary_dim

        #embedding layer
        self.embedding = nn.Embedding(vocabulary_dim, embedding_dim, padding_idx=0)

        #lstm layer
        self.lstm = nn.LSTM(embedding_dim, hidden, num_layers=layers)

        #classifier
        self.linear = nn.Linear(hidden, output_dim)

        #softmax
        self.softmax = nn.Softmax(dim=-1)



    def forward(self, x, h=None, c=None, softmax=False):

        '''
        Softmax is a Boolean that when training is false since the CrossEntropy Loss Function will be called (which has Sofmax already built in)
        but will be set to true when we call the network for text generation, and thus we would need a probabitu distribution
        '''

        emb = self.embedding(x)

        if h is not None and c is not None:
            out, (h, c) = self.lstm(emb, (h, c))
        else:
            out, (h, c) = self.lstm(emb)

        out = out.contiguous().view(-1, self.hidden_dim)

        x = self.linear(out)

        if (softmax):
            x = self.softmax(x)

        return x, h, c

'''Ex 1.3.2: implement the text generator '''
def text_generator(characters, net, longtextData, initial_text, mode=0):
    '''
    Generate text starting from an initial text

    Args:
        :param characters: number of characters to predict
        :param net: NN
        :param longtextData: object containing the vocabulary
        :param initial_text: initial text from which to start
        :param mode:
                if mode == 0: greedy choice of the new character (choice with argmax)
                if mode == 1: sampling choice

    Return:
        :return: res = initial_text + predict text
    '''
    #initial_test must be set
    assert initial_text

    res = initial_text
    net.eval()
    with torch.no_grad():
        h, c = None, None

        #update the state of the LSTM considering the input string
        ids = np.zeros(shape=(len(res),1))

        #convert char to ids
        for char in range(len(res)):
            char_string = initial_text[char]
            ids[char] = longtextData.vocab.string_to_id[char_string]

        #convert np to tensor
        id_char = torch.from_numpy(ids).to(device)
        id_char = id_char.type(torch.int32)

        #update state of the net with the given text
        for i in range(id_char.shape[0]-1):
            probs, h, c = net(id_char[i].view(1,-1), h, c, softmax=False)

        last_char_id = id_char[-1,0]

        for p in range(characters):

            prob, h, c = net(last_char_id.view(1,-1), h, c, softmax=True)

            # greedy choice
            if (mode == 0):
                value, indices = torch.topk(prob.detach(), k=1, dim=-1)
                new_ch = indices[0].item()

            # sampling choice -> Ex 1.3.3: implement the sampling choice
            elif (mode == 1):
                value = torch.multinomial(prob.detach(), num_samples=1)
                new_ch = value.item()

            res += longtextData.vocab.id_to_string[new_ch]
            last_char_id = torch.LongTensor([new_ch]).to(device)

    return res

'''Ex 1.3.4: implement the code for training'''
if __name__ == '__main__':
    print('\n \t--- START TRAINING --- \t \n')
    loss_function = torch.nn.CrossEntropyLoss()
    # input just a single character

    # other hyperparameters
    embedding_dim = 64
    hidden_dim = 2048
    n_layers = 1
    initial_text = "Dogs like best to "
    characters_to_predict = 10 # for the ex 1.3.7 I used characters_to_predict = 50
    mode = 0  # greedy -> change this to 1 to test point 1.3.8

    # init net
    net = Net(my_data.vocab.__len__(), my_data.vocab.__len__(), embedding_dim=embedding_dim, hidden=hidden_dim,
              layers=n_layers).to(device)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    h, c = None, None

    # target tolerance
    tol = 1.03

    # deatch h and c for the second batch
    first = 1
    epoch = 1
    step = 0
    max_epoch = 100
    continue_training = 1

    perplexity_vec = []
    generated_text = []
    while (continue_training):
        perplexity_mean = 0
        for batch in batches:
            net.train()

            # truncate propagation for each batch
            if not first:
                h = h.detach()
                c = c.detach()

            # reset grad
            optimizer.zero_grad()

            # reshaping target
            target = batch[1:, :]
            target = target.reshape(-1)

            probs, h, c = net(batch[:-1, :], h, c)

            # compute loss
            loss = loss_function(probs, target)

            # compute perplexity
            perplexity = torch.exp(loss).item()

            perplexity_mean += perplexity

            loss.backward()

            # clipping
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            # update params
            optimizer.step()

            if first:
                first = 0

            if (perplexity < tol) or (epoch > max_epoch):
                continue_training = 0

            if (step % print_step == 0):
                res = text_generator(characters_to_predict, net, my_data, initial_text, mode=mode)
                print(res, '\n')
                print('Perplexity: ', perplexity)

            step += 1

        generated_text.append([epoch, res])
        epoch += 1
        perplexity_mean = perplexity_mean / len(batches)
        perplexity_vec.append(perplexity_mean)


    print(generated_text)

    steps_vec = np.arange(len(perplexity_vec))
    plt.plot(steps_vec, perplexity_vec)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Perplexity per Epoch")
    plt.title("Perplexity")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,16))
    plt.show()


    print('done')