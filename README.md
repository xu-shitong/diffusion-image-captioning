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
  - Try use 16-256 dimention embedding, 
    - diffusion lm used 800-16 dim embedding and linear project 16 to 768 for bert dim
    - run 200 diffusion on college gpu
      - performance significantly better than 3 sample
    - Running 100 diffusion with both clip and text sample
      - performance ok, to be compared with 200 way
    - TODO: self defined tokenizer not working
      - abandon for now, TODO in later in abliviation
    - use kl loss instead of mse loss/check how diffusion lm use mse
2. combining image feature
  - image feature as time embedding add to transformer
    - check how to use same tokenizer as transformer
    - due to clip has 512 dim, use distilbert with 512 dim and 8 head, like transformer instead of bert
    - todo: add linear layer and add feature elementwise
  - model input [x_t ... x_t, image_clip, text_clip], output at position [x_t ... x_t] is model prediction
    - only diffusion model, 
      - train use all info to get model regenerate text
        - again produce reproducing tokens with 10e5 rounding weight.
        - should not change embedding and head number, otherwise only capable of reproduce in low x_t
      - try reproduce with new tokenizer and embeddings
        - todo: mask wrongly unchanged, change and retrain
        - abandoned, should not change dim
      - train use and not use image clip to get un classification guided model, 
        - not use classification guided, by mask image clip sometime
        - use classification guided method:
          - changing func is linear for each position
          - changign func is transformer to [x_t, x_t] part
  - todo: clip as additional feature vector to sequence, add segment 