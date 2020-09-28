import torch
from LanguageModel.NNLM import NNLM
import torch.nn as nn
import torch.optim as optim

"make_batch： 对于每一个sentence,将最后一个单词当作target,前面所有单词当作input"
def make_batch(sentences):
    input_batch = []
    target_batch = []
    for sen in sentences:
        word = sen.split() # space tokenizer
        input = [word_dict[n] for n in word[:-1]] # create (1~n-1) as input
        target = word_dict[word[-1]] # create (n) as target, We usually call this 'casual language model'
        input_batch.append(input)
        target_batch.append(target)
    return input_batch, target_batch
if __name__ == '__main__':

    n_step = 2 # number of steps, n-1 in paper
    n_hidden = 2 # number of hidden size, h in paper
    m = 2 # embedding size, m in paper
    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}

    vocab_size = len(word_dict)  # number of Vocabulary

    "构造NNLM模型,声明损失函数和梯度下降方法"
    model = NNLM(vocab_size,m,n_hidden,n_step)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    input_batch, target_batch = make_batch(sentences)
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)
    print(input_batch.size(),target_batch.size())

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])