---
title: "MarkupLM- LayoutLM for HTML Pages"
collection: talks
permalink: /talks/MarkupLM
date: 2023-06-17
---

<p align="center" width="100%">
    <img width="50%" src="/assets/images/markuplm_img1.png">
</p>

Multimodal pretraining with text, image, and layout made progress in recent times. One uses 2D positional information i.e. BBoxes to identify the position of text in the layout. For the Large no of digital documents with consistent changes in layouts, existing models are not working. 

Authors proposed MarkupLM, where both the **text** and **markup** information is pre-trained. While the layout and style of Markup Pages change with respect to Devices the pages opened. with frequent changes in layout in different devices, fixed layout-trained models don't perform well. 

We need to make use of the Markup Structure in the Documents for these documents. MarkupLM uses different units(tags in standard HTML) available in Markup Pages to identify the position, unlike bboxes in pdfs.

#### Embedding layers:
MarkupLM with 4 input embedding layers:
- text embedding <-- represents tokens & their sequence information.
- Xpath(XML path) embedding <-- represents markup tag & their sequence information from RootNode to CurrentNode
- 1D position embedding <-- represents only sequence order information.
- Segment embedding <-- for Downstream tasks.

![](../assets/images/markuplm_img1.png)
The Xpath(XML path) embedding layer herein above is the replacement for the 2D position embeddings(bbox) in the layoutLM model. 

#### Pretraining Strategies:
The authors used 3 pretraining strategies for effective training:
- Masked Markup Language model(MMLM): to learn joint contextual information of text & markup
- Node Relationship Pretraining(NRP): to define relationships according to the hierarchy from the markup trees. (check the dom tree in the above figure)
- Title Page matching(TPM): Randomly replacing the Title from another page to make the model learn whether they are correlated.

Using all the above strategies MarkupLM can better understand Contextual information through both text and Markup.

#### Model:
BERT as the Encoding backbone & later replaced it with Roberta and significant performance change was observed.

#### Pretrianing:
We initialize MarkupLM from RoBERTa and train it for 300K steps. We set the total batch size as 256, the learning rate as 5e-5, and the warmup ratio as 0.06. The selected optimizer is AdamW with = 1e − 6, β1 = 0.9, β2 = 0.98, weight decay = 0.01, and a linear decay learning rate scheduler with 6% warmup steps. We also apply FP16, gradient checkpointing and Deepspeed to reduce GPU memory consumption and accelerate training.

### Reference:
- https://aclanthology.org/2022.acl-long.420.pdf
- https://huggingface.co/docs/transformers/model_doc/markuplm#markuplm




