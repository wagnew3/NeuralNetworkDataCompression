## Data Compression Using Neural Networks
The [Hutter Prize](http://prize.hutter1.net/) measures an intelligence by its ability to compress the first 100MB of Wikipedia text. Just as all of human mathematics could be derived from a small set of axioms by a sufficiently intelligent human, the idea is that a more intelligent AI will be able better understand the Wikipedia text, and so describe it using a smaller set of rules.
Deep neural nets, in varied architectures, are responsible for many of the most impressive recent advances in AI-they dominate computer vision and NLP. So, over about a month, I tried pitting them against impressive human crafted heuristics in the area of data compression.

##Neural Networks Used
* GRU Networks
* LSTM Networks
* [MemN2N](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf), as implemented by [Taehoon Kim](https://github.com/carpedm20/MemN2N-tensorflow)

## Results
I was unable to win the Hutter Prize (for now, at least).
Major conclusions: GRU networks vastly outperform LSTM networks. Don't train the network to predict rare characters: this greatly increases network size, and the network is unlikely to predict these characters very well in any case.
Many networks were easily able to drop below a training error of 1.1 (Cross Entropy), which (in theory) means that they could reproduce the Wikipedia text with only 13.75MB of "correcting" information-enough to win the current Hutter Prize! 
However, I was unable to find a good way to go from a network encoding with 1.1 cross entropy to a compressed image of the Wikipedia text of size 13.75MB. Even when treating the neural network as a preprocessing step, replacing the Wikipedia text with the ranking the network's predictions of the correct character, and then compressing this file with previous Hutter Prize winners did not produce good results, perhaps because these compression algorithms as specialized to Wikipedia text, and not the network's errors on the Wikipedia text.
In addition, the weights of the neural networks themselves are quite large: a FP32, 5-layer GRU with 250 neurons in each layer is around 4MB. This challenge makes the problem much more interesting: the ML scientist may look at this problem and dismiss it as encouraging overfitting and so of little practical relevance. However, overfitting is just as harmful: to get the most out of a neural network of a certain size, the most generally applicable rules must be learned. Indeed, this problem provides a way of comparing learning architectures that is not obvious from traditional scenarios where neural network size is not a great concern: how well an architecture can learn for its size, or, bluntly, learning density.

##Things I (You?) Will Try
* More neural networks that use memory, like Google Brain's [Differntiable Neural Computers](http://www.nature.com/nature/journal/v538/n7626/full/nature20101.html)
* Modifications to neural networks that greatly reduce the space they take: [Binarized Neural Networks](https://arxiv.org/pdf/1602.02830.pdf), [Compressing Neural Networks Using Hashing](http://www.jmlr.org/proceedings/papers/v37/chenc15.pdf) (and several other techniques presented in papers discussed in the introduction)
* Improving/finding a graph computing library to allow for recurrent neural networks to be unrolled thousands of times