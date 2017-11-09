# BiMPM_Pytorch
Pytorch implementation of Bilateral Multi-Perspective Matching [1] using in [Quora Question Duplicate Pairs Competition](https://www.kaggle.com/c/quora-question-pairs). You can find the original tensorflow implementation from [here](https://github.com/zhiguowang/BiMPM). 

## Description

`models/model.py` - model graph.

`models/MultiPerspective.py` - multi perspective matching layer.

`train_bimpm_gpu.py` - train and test BiMPM model.

If you find any bugs, please create an issue, thanks.

## Requirements

- python 3.6
- pytorch 2.0
- numpy 1.12.1
- pandas 0.19.2
- nltk 3.2.2
- gensim 1.0.1
- argparse

## References

[[1]](https://arxiv.org/pdf/1702.03814) Zhiguo Wang, Wael Hamza and Radu Florian. "Bilateral Multi-Perspective Matching for Natural Language Sentences."
