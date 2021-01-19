## Overview
It is reimplementation of [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/abs/1502.04623 "DRAW: A Recurrent Neural Network For Image Generation") to generate images of cats using this [Cats dataset](https://www.kaggle.com/crawford/cat-dataset "Cats dataset")
Code used to cut heads of cats and make some data augmentation is [here](https://github.com/aleju/cat-generator/tree/master/dataset "here").

The code is based on the work of [Eric Jang](https://blog.evjang.com/2016/06/understanding-and-implementing.html "Eric Jang") and [Samuel Noriega`s code](https://medium.com/3blades-blog/draw-a-recurrent-neural-network-for-image-generation-725b39ef824f "Samuel Noriega'c code"), who cleaned it up.
We are also in the process of adjusting the code to the Tensorflow 2.x, because the original implementation was done in Tensorflow 1.x.

## Results of reimplementation in Tensorflow 2.x
| Without Attention RGB  | With Attention Gray |
| ------------- | ------------- |
| <img src="https://imgur.com/6TOkMFC.gif" width="70%" class="center"> | <img src="https://imgur.com/nVKaYhw.gif" width="70%" class="center"> |
