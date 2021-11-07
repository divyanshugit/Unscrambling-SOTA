# Word2Vec:

Gist
---
In this paper, researchers proposed two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. It shows that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities.
 
It was observed that it is possible to train high quality word vectors using very simple model architectures, compared to the popular neural network models (both feedforward and recurrent). IT outperforms the previous state of the art is the [SemEval-2012 Task 2](https://www.academia.edu/13078692/Semeval_2012_task_2_Measuring_degrees_of_relational_similarity).

Original Paper:[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

Blog by [Jay Alammar](https://github.com/jalammar): [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)

```python
# Inspired from(Credit): graykode

if __name__ == '__main__':
    batch_size = 2 
    embedding_size = 2 

    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    # Make skip gram of one size window
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()   

```
Check out the notebook for more info: [word2vec.ipynb](https://github.com/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec-Skipgram(Softmax).ipynb)