---
title: "Attention is All you Need - Birth Of Modern AI"
collection: talks
permalink: /talks/attentionisallyouneed
date: 2023-06-03
---

This is more of my understandings about Transformers rather than the paper summary.

The paper that revolutionized the Modern Deep learning both in computer vision, Natural language processing, audio, speech and more. These were first introduced for Language translation.

### Why not Traditional NLP:
RNNs, LSTMs, GRUs are used before transformers & best fits for seq-seq prediction. Check this [post](https://purnasai.github.io/Recurrent-Neural-Networks-and-LSTM-explained/) for more about RNNs. 

RNNs are Slow. They use time step at every token computation. Their gradients vanish with time or explode due to chain computations due to long network which makes them to retain only recent information. the gradients are not retained for longer text. They are not parallelizable.

Usage of Truncated BTT(ex: using a window of last 10 timestamps only) also makes the model too slow than before and stills fail to handle longer text. 

Usage of **keep | Delete | update gates** in LSTM makes sequence processing better than RNNs. But even LSTMs are also too slow than RNNs as they involve heavy computation inside gates.

### Modern NLP:


![](../assets/images/attention_img1.png)

#### Step1: Input Embeddings
input Text/sequence --> each word is tokenized in to sub tokens(like BERT using word piece tokenization, so we can match most text, even if it is not in learned text vocabulary.) --> each sub token assigned a number(input ID) from the vocabulary --> each input ID is again given an embedding of 512 dimensions. 

#### Step2: Positional Encoding:
it represents a pattern. sin & cos trigonometric functions are used to add a positional information to the input embeddings. this lets the model to know the position of each word in entire sentence.

![](../assets/images/attention_img2.png)

#### step3: Attention(Scaled dot product attention)
Lets say the given input text is query(Q) with dimensions 10x512, representing 10 words and each word with 512 dimension embeddings. it is multiplied with K^T, i.e Key Transpose of 512x10, gives 10x10 after multiplied. The Dot product is applied here. 
![](../assets/images/attention_img3.png)
This dot product is divided by square root(512) or Scaled & multiplied with Value again, gives Attention mechanism.

 ![attn | left | 300x300](../assets/images/attention_img4.png)
 We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values. This dot product tells which word embedding is close to other word embedding, higher the score higher the agreement. 
 
 Dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code. 


![im | 500x400](../assets/images/attention_img5.png)


we get weights from scores that we got from dot product. we apply softmax and normalize them.

Why do we multiply with values?
because values are original word vectors and we multiply them with result from softmax, which are weights that were learned from softmax & dot product of q,k. So when we multiply W with V again, we are attaching weight information to V.  This step is what that introduces context.

Attentions is a function similar to a "Soft Lookup" in (Query, Key, Value) Dictionary.
![](../assets/images/attention_img6.png)
More to it:
![](../assets/images/attention_img7.png)

We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4 . To counteract this effect, we scale the dot products by √dk .

#### Step4: Multi-head Attention
all the 3 (Q, k, V) are multiplied with respective Weights(W), that are trained/learnable parameters gives W prime. So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three weight(w) matrices that we trained during the training process.

The dimensionality of weight matrix is 64, while input dimension is 512. this is an architectural choice.  Now we split these 3 prime weights (Q,K,V) in to parts by dimensions. so that all the 4 heads see the part of embedding of each word, which means each head will see full sentence, but different aspect of the same words w.r.t to other words.

![](../assets/images/attention_img8.png)

#### Encoder:
stack of N=6 identical layers. 2 sublayers again. MHSA(Multiheadselfattention)+ FFN(Fullyconnectedfeedforward network). A skip connection for these 2 sublayers with Layer norm. Encoder output is a heavily processed, high level, contextualized version of input text. this has nothing to do with the requested output yet & is only half part of the game.

#### Decoder: 
A stack of N = 6 identical layers. A third sub-layer, which performs multi-head attention over the output of the encoder stack. We modify the self-attention sub-layer to Masked self attention that prevent positions from attending to subsequent(future) positions. 

This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

#### Position wise feed forward network:
The second sublayer in encoder and decoder blocks.

![](../assets/images/attention_img9.png)

#### Applications of Attention in the model:
- Queries come from the previous decoder layer & Memory keys and values come from the output of the encoder. 
>This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models

- A self-attention layer in encoder with all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. 
>Each position in the encoder can attend to all positions in the previous layer of the encoder.

- Self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder **up to and including that position**. 
> We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections.


All made these components powerful:
- complete Attention
- positional encodings with sin, cos trigonometry functions.
- residual/skip connections doesn't let model gradients vanish.
- interspersing of attention and MLP
- layers norms
- multiple heads attention<- parallelized, reduced computation, context calculation.

Various Versions of Transformers:
![](../assets/images/attention_img10.png)

This will be continuously updated as I learn through.

Most of the Images used in this post are from the different resources available.

#### Thank you

A PPT presentation by Lucas Beyer [here](https://docs.google.com/presentation/d/1ZXFIhYczos679r70Yu8vV9uO6B1J0ztzeDxbnBxD1S0/mobilepresent?slide=id.g13dd67c5ab8_0_3897).
https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy
https://www.youtube.com/watch?v=QCJQG4DuHT0&ab_channel=CodeEmporium
https://www.youtube.com/watch?v=XfpMkf4rD6E&ab_channel=StanfordOnline
https://www.youtube.com/watch?v=t5LjgczaS80&ab_channel=FormalLanguagesandNeuralNetworksSeminar

Encoder block excellently explained here:
https://www.youtube.com/watch?v=g2BRIuln4uc&ab_channel=Ark

best and indepth:
http://nlp.seas.harvard.edu/2018/04/03/attention.html
https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html



