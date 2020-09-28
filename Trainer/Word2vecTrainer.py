import numpy as np
from LanguageModel.word2vec import Word2vec
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

def random_batch(skip_grams):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)
    for i in random_index:
        #用np.eye函数将输入转换为one-hot编码
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target
        random_labels.append(skip_grams[i][1])  # context word
    return random_inputs, random_labels

if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2  # mini-batch size
    embedding_size = 2  # embedding size
    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    "word2id,id2word"
    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    # 当context window size 为1时
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])
    print(word_dict)
    print(len(skip_grams))
    model = Word2vec(voc_size,embedding_size).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)



    # Training

    for epoch in range(5000):

        input_batch, target_batch = random_batch(skip_grams)

        input_batch = torch.Tensor(input_batch).to(device)

        target_batch = torch.LongTensor(target_batch).to(device)



        optimizer.zero_grad()

        output = model(input_batch)



        # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)

        loss = criterion(output, target_batch)

        if (epoch + 1) % 1000 == 0:

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))



        loss.backward()

        optimizer.step()


    print(word_list)
    weights=[]
    for p in model.parameters():
        weights.append(p)

    wordvec=weights[0]
    for i, label in enumerate(word_list):


        x, y = wordvec[0][i].item(), wordvec[1][i].item()

        plt.scatter(x, y)

        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    plt.show()

