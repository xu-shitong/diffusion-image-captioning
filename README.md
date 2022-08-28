# diffusion-image-captioning

Research project on image captioning using diffusion language model

1. get distilBERT capable of generating sentence by using diffusion model
  - use single sample trials
    - no training embedding and projection, due to model might collapse
    - x_1 and x_0 are almost same, but even linear layer fail to learn identity
    - pretrained projection layer with no bias is better, since input used no bias
    - adding embedding loss greatly improve performance, first model manage to predict without repeat word
  - use multiple samples
    - predict x_0
      - rounding error 0.05
        - use 3 sample, 3 epoch, no embedding training, l1 recreate loss 0.078 but fail to reproduce x_1 and x_5
        - use 2 sample, 2 epoch retrain, l1 loss 0.088 also fail to reproduce x_1 x_5
        - use 1 sample, 3 epoch, to proof sampling has error, still fail reproduce
      - rounding error 0.5, use 3 sample, 3 epoch, no embedding training, l1 recreate loss 0.568 correctly restore till x_20, multi sample working
    - predict x_{t-30}
      - rounding error 0.1, 3 sample, 1 epoch, model reproduce with poor performance
      - rounding error 0.1, 3 sample, 3 epoch, l1 loss 0.1, only briefly reproduce x_1
      - rounding error 0.3, 3 sample, 3 epoch, l1 loss 0.5, could restore most info till x_30, increase rounding error does not further improve performance
    - try predict noise not x_0, abandoned, due to predicting z result in generation of only x_t-1, mathematically not capable of generating multi steps later latent
    - try cosine noise scheduling
      - todo experiment
    - try language use improved model, with resampleing
    - todo: try time embedding to transformer
2. combining image feature
  - image feature as time embedding add to transformer
    - check how to use same tokenizer as transformer
    - due to clip has 512 dim, use distilbert with 512 dim and 8 head, like transformer instead of bert
    - 