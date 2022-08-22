# diffusion-image-captioning

Research project on image captioning using diffusion language model

1. get distilBERT running
2. get distilBERT capable of generating sentence by using diffusion model
  - no training embedding and projection, due to model might collapse
  - x_1 and x_0 are almost same, but even linear layer fail to learn identity
  - pretrained projection layer with no bias is better, since input used no bias
  - adding embedding loss greatly improve performance
3. combining image feature