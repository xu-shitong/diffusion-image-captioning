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
what will happen if no clip is given
3. use coco dataset, todo
4. use validation dataset, training
5. try directly allow output longer string, fail, always have sep at the end, might due to dataset always end with sep at 16th position, model overfit
6. data argumentation might be helpful
7. use clip supervised method, left over work
CIDEr metrics


65835: epoch15_lossseries_sum_lr5E-05-5E-08_schedulerlogspace_round1E+00_dynamic1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue: 前几epoch代价应该会上升，观察从哪一epoch代价下降
- 代价值一直没有下降

65840: epoch5_lossseries_sum_lr5E-07-1E-08_schedulerlogspace_round1E+00_dynamic-1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue: 观察代价值是否下降
- 代价值维持在 1100+ 11.35， prob一直从10000+ 下降到9900+

epoch5_lossseries_sum_lr5E-07-1E-07_schedulerlogspace_round1E+00_dynamic2_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue：尝试增大相对ratio到2

epoch5_lossseries_sum_lr5E-07-5E-07_schedulerlogspace_round1E+00_dynamic2_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue：尝试重新得到dynamic = false时的5e-7代价值下降的例子
- 代价值下降，但800不如5e-5 dynamic 2的300

epoch5_lossseries_sum_lr5E-07-1E-07_schedulerlogspace_round1E+00_dynamic2_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue：尝试基于5e-7得到schedule learning rate的成功例子
- 代价值下降小于直接固定使用5e-5，lr减小太快

epoch5_lossseries_sum_lr5E-05-3E-05_schedulerlinspace_round1E+00_dynamic1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue：dynamic 1 + 维持lr5e-5 效果较好 在300+ x_t 1500+ prob
- x_t代价值300+ 增长至400+ prob代价值下降

epoch5_lossseries_sum_lr5E-05-5E-05_schedulerlinspace_round1E+00_dynamic1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue：尝试直接维持lr在5e-5
- x_t代价值300+ 增长至400+ prob代价值下降

epoch15_lossseries_sum_lr5E-05-5E-05_schedulerlinspace_round3E-01_dynamic-1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue：重新得到原模型样例

epoch5_lossmse_series_sum_lr5E-05-5E-05_schedulerlinspace_round1E+00_dynamic-1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue：mse做代价函数
- 单样本x_t代价值32，x_0代价值64 prob从130上升到200，无法准确预测x_1

epoch5_lossseries_sum_lr5E-05-5E-05_schedulerlinspace_round1E+00_dynamic2_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_tFalse_use_x_1False_use_probTrue：尝试仅使用prob loss，loss 
- prob到3.43，后epoch变为nan，模型没有converge

尝试5 epoch series sum并每一loss / 100，观察是否为权重导致学习不佳，不使用dynamic，rounding 为1，lr固定5e-5
- x_1代价值回升，x_t缓慢下降
- 尝试改回0.3 dynamic weight

尝试5 epoch epoch内log从5e-5降至5e-7，使得第三epoch的lr为5e-6，避免loss增加。dynamic 为1
- 本地实验为prob loss在第二epoch后上升
- x_1代价值略微上升 另外代价值下降，尝试扩大训练

epoch15_lossseries_sum_lr5E-05-5E-05_schedulerlogspace_round3E-03_dynamic-1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_tTrue_use_x_1True_use_probTrue：尝试5 epoch series sum并每一loss / 100，观察是否为权重导致学习不佳，不使用dynamic，rounding 为0.3，lr固定5e-5
- 

epoch15_lossmse_series_mean_lr5E-05-5E-05_schedulerlogspace_round3E-01_dynamic-1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_tTrue_use_x_1True_use_probTrue：尝试对mse loss 取batch的mean，seq和dim层面取mse round 0.3，lr 5e-5 不使用dynamic
- 

epoch15_lossseries_sum_sample_mean_lr5E-05-5E-05_schedulerlogspace_round2E-01_dynamic-1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_tTrue_use_x_1True_use_probTrue
- 初步检查能够converge，x_t上升，
- 测试bleu

epoch15_lossseries_sum_sample_mean_lr5E-05-5E-05_schedulerlogspace_round1E+00_dynamic-1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_tTrue_use_x_1True_use_probTrue
- 能够converge，所有代价值下降，错误识别人狗
- 测试bleu

epoch15_lossseries_sum_sample_mean_lr5E-05-5E-05_schedulerlogspace_round7E-1_dynamic-1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_tTrue_use_x_1True_use_probTrue
- 尝试0.5 rounding weight
- x_t代价回升后下降，总体效果不如0.3

epoch5_lossseries_sum_sample_mean_lr5E-05-5E-05_schedulerlogspace_round3E-01_dynamic-1_clipconcat_clipmaskNone_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_tTrue_use_x_1True_use_probTrue
- 尝试0.3rounding + guidance 
- x_t代价值有上升趋势，在4.8左右，尝试再训练10 epoch得到最终效果
- 最终仍不如当前最优
  
epoch5 round 0.3 使用或不使用guidance, 5e-5 lr
- 尝试是否当前代码存在bug，导致模型代价值会上升一段
   可能由于使用较大数据集，没有代价上升

尝试5 以上设置并epoch内log从5e-5降至5e-7，使得第三epoch的lr为5e-6，避免loss增加。dynamic 为1
- 可能由于使用较大数据集，没有代价上升

14 epoch 30+8数据集 guidance
- abandon, classifier free 实现错误

15 (5 + 10) epoch 8数据集 0.3 no guidance 得到小数据集的最优模型效果
- 记录为baseline简略版

1e-4 lr
- 实验效果好于baseline

1e-5 lr
- 15epoch 后仍有下降趋势，预计效果不如1e-4

dynamic weight 5e-5 0.3
- 不应为0.3，取3，由于weight为1时在最后几epoch 倍率为3-5。3实际对应rouding weight维持为1

10 epoch 尝试3倍率dynamic 若效果好继续训练后5 epoch
- 代价值较高但生成效果接近大数据集训练结果，继续训练5 epoch
- 效果不如baseline

尝试无dynamic 5e-4
- prob部分converge失败

(添加新classification free 代码

尝试15 epoch 无guidance lr linear 降低1e-4 - 1e-5
- 

尝试5 epoch guidance free
- 能够converge，效果暂时差于1e-4最佳结果
- 额外增加5epoch

尝试embedding

尝试add 而不是concat


基于5e-5的前两epoch结果尝试下一lr应取的值


证明5e-5时dynamic 为1 最优：
  epoch15_lossseries_sum_lr5E-05_round1E+00_dynamicFalse_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue

  epoch5_lossseries_sum_lr5E-05-5E-05_schedulerlinspace_round1E+00_dynamic1_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue

  epoch5_lossseries_sum_lr5E-05_round1E+00_dynamic2_clipconcat_clipmask10_train-embedFalse_samplesize100_x_0_predictTrue_X_INTERVAL100_use_x_1True_use_probTrue