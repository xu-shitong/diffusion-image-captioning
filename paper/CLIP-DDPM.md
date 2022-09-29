# Abstract
In this work, we applied denoising diffusion probabilistic models to text generation in image captioning tasks. We show that our CLIP-diffusion-LM *** On the flickr8k dataset, the model showed. By combining samples from flickr8k and flickr30k dataset, our model showed ... performance. In addition, the model achieved ... zero shot performance in COCO 2015 image caption task. Our code is available at ...


contribution: experiment on learning rate, optimizer, adding mechanism, cosine schedule, using classifier free or not. using model to predict image feature or not, scale up to larger dataset

# Introduction
Image captioning has being a focus of research over the recent years. Previous text encoder used could be split to 2 general classes, i.e. `autoregressive` and `non-autoregressive`. Most of the sota models falls in the `autoregressive` class[]. However, autoregressive generation suffer from 1) the slow generation speed, because the generation is token by token, and 2) not capable of refining prefix of sentences based on the later generated tokens. Multiple attempts have experimented using a non-autoregressive model in the text generation steps[]. The closest to our work is 2019 Masked Non-Autoregressive Image Captioning[], which used a BERT model as generator and involves 2 steps to refine the generated sequence. However, these work still used a discrete generation process, which means masking out certain tokens and train model to refine words in these certain positions. To the best of our knowledge, there has not been other work on generating caption embedding based on continuous generations steps. Our work aim at employing a model to refine generated token continuously on their embedding. In particular, we used pretrained CLIP model for extracting image and text features, and distilbert model based on diffusion-lm for text sequence generation. Our contribution could be summarized as follow:
  - apply diffusion model in image captioning tasks 
  - experiments with multiple feature fusion methods, in particular the relative importance of restoring the token feature and certainty of generated sequence.


(link to each experiments)


# Related work

## Autoregressive image captioning
2015 deep caption with multimodal rnn[] proposed the mRNN model, which used CNN for image extraction and RNN for text generation. 2016 show attend tell[] employed the LSTM for text generation and exerimented on soft and hard attention for early fusion of image feature and text feature. Based on this, 2016 knowing where to look[] experimented the late fusion of image and text features, allowing model to attend on either image or text modality. 2017 cascade recurrent nn[] experimented on reversing the generated caption, allowing its backend model to refine the former tokens based on later caption tokens. 2018 gla[] used attention model to combine local and global feature from images, so that captions can more accurately identify occluded objects. Similarly, 2019 stack vs[] also used image features from both high and low generaliaty, and combined them using cross attention. Their work also involves multi step refining of the generated text caption. Caption in each step the caption is autoregressively generated. 2019 unsupervised image caption[] trained image caption in a GAN style, with a LSTM discriminator reproducing the original image feature from generated text sequence. 2019 Variational Autoencoder-Based Multiple ImageCaptioning Using a Caption Attention Map[] used variational auto encoder for extracting image information, their model allows(?) changing the image feature by sampling from the learned distribution, thus produce various captions for different images. 2021 CLIP cap[] experimented on using pretrained CLIP image feature for sequence generation. The CLIP features are transformed to a sequence of token and used as prefix for a GPT-2 model in generation. 
2022 GRIT[] experimented changing the cross attention part of transformer decoder to use both Reginal feature from Faster RCNN and Grid features from swin transformer. 

## Non autoregressive image captioning
In contrast to autoregressive generation, non-autoregressive models benefit from the models' ability to pass textural information in both direction during generation. The text generated in former timesteps could adjust based on text in later timesteps, thus is expected to achieve better performance. 2019 Masked Non-Autoregressive Image Captioning[] used BERT[] as text decoder and employed a 2 step generation method. Based on this work, Partially Non-Autoregressive Image Captioning [] and semi Non-Autoregressive Image Captioning[] partitioned the generated text in subgroups, words in the same group are generated non-autoregressively and different groups are generated in autoregressive way. Our method falls in this catogory and most close to the 2019 Masked Non-Autoregressive Image Captioning[]. The difference is we chose to use diffusion model as the non-autoregressive generation model.

## Diffusion models
[] proposed the DDPM model to simplified the loss function by only letting models to predict the noise in generation steps, and proposed alternative loss functions by removing the weight terms. Based on DDPM, improved ddpm [] proposed several improvments based on DDPM, including setting variance to be learnable parameters, apply cosine instead of linear noise schedule, and speed up forward process by reducing forward steps. 

Diffusion model beat GAN[] is another work which used larger model, more attention heads, multiscaled attention and classifier guidance(?is the classifier guidance reduce generation step or higher score) for reducing generation steps. Classifuer guided diffusion is a technique of using a classifier to help generating features with higher . The classifier is pretrained on dataset with noise added. To guide the generation, the model's output is feed into the classifier, to get gradient on which direction to optimise parameter in order to generate image closer to target class.

To avoid training classifier for guiding model, classifier free guidance technique is proposed, the task is to provide generator both guided or unguided samples, and train an additional model to predict the relationship in 2 models' output. Example of applying classifier free guidance includes GLIDE[], (other class free guide work?)

Diffusion lm[] which is a recent work on applying continuous diffusion model on text generation, this paper provides various techniques to improve the performance of continuous diffusion model on text generation. 

ddim[] reduced the variance in forward process. The result showed that by reducing variance to 0, the deterministic model achieved higher FID score in image generation on both CIFAR10 and CelebA. 

By using diffusion model as text-to-image generation, DALL-E 2 and GLIDE model achieved significant image generation performance. Dall-e 2[] is a recent work on using CLIP and diffusion model for image generation task. The model used CLIP model for extracting feature from text, predict the corrisponding image CLIP feature through prior network, then use predicted image CLIP feature for final image generation. The model achieved significant novelty in generated images. The innovativity of generated of image from DALL-E 2 also provided us the inspiration to train a image-to-text model with diffusion model in generation step. 

# Background

## Diffusion models
The training of denoise diffusion probabilistic model involves generation of noised samples (forward process), and denoising based on model's output (backward process). Let x_0 be the original feature, the forward process incremently add noise to $x_0$ to generates a sequence of T noised features $[x_1, ... x_T]$. Each x_t at step t is sampled from probability distribution $q(x_t | x_{t-1}) = N(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$. From reparameterization trick, the x_t at any step t could be directly generate from x_0: $x_0 = ...$. The backward process is using a trained model with parameter $\theta$ to denoise the samples generated in the forward process. The training objective is to minimize the negative log-likelihood of generating x_0, that is to minimize $E(-\log(p_{\theta}(x_0))) = E(-\log(\int p_{\theta}(x_0, ..., x_T), d(x_1, ..., x_T))) = E(-\log(\int p_{\theta}(x_0, ..., x_T), d(x_1, ..., x_T)))$. Where $p_{\theta}(x_{t-1} | x_t) = N(x_{t-1}, \mu_{theta}(x_t), ...)$ and $\mu_{\theta}$ is model's prediction on mean of $x_{t-1}$ conditioned on $x_t$. By modeling the backward process as a Markov Process, $p_{\theta}(x_1, ..., x_T)$ is simplified to $p_{\theta}(x_T) \prod_{t=1}^T p_{\theta}(x_{t-1} | x_t)$. From variational lower bound, $E(-\log(p_{\theta}(x_0))) \leq E_q[\log(\frac{p_{\theta}(x_0..x_T)}{q(x_1..x_T| x_0)})]$ $= E_[\log(p(x_T)) + \sum_{t = 1}^T\log(\frac{p_{\theta}(x_{t-1} | x_t)}{q(x_t | x_{t-1})})]$. From the work of [], expanding and reweighting each term of the negative log-likelihood gives a concise loss function $L = \sum_{t=1}^T E_{q(x_t | x_0)} \|\mu_{\theta}(x_t, t) - \mu_{x_t, x_0}\|^2$

Due to the large generation step number (T = 1000 as proposed in []), and the generation step being autoregressive on the denoised feature in the previous step, the reverse diffusion is significantly slower than the other generative models (... for gan and ... for diffusion). Multiple stetegies were proposed to accelerate the generation process. In Improved DDPM [] a subset of generation steps is selected. Model is trained to predict ... . In diffusion-lm the model is trained directly predict the x_0 instead of the intermediate steps containing noise. In our experiments, the ... showed better reproduce quality compared with... {}. In addition, by following the x_0 prediction method, our model could converge to a reasonable output in less than 5 diffusion steps. This step number is which is significantly less than autoregressive methods using an encoder-decoder structure. , which has generation steps propotional to the output sequense length. autoregressively apply the output of x_0 trained model further improved the performance. {}

Based on the propose from, we added an additional rounding term[] in our loss function, parameterized by $E_{p_{\theta}(\hat{x} | x_t)}-\log(p_{\theta}(w | \hat{x})) = E_{p_{\theta}(\hat{x} | x_t)}-\log(\prod_{i=1}^L p(w_i | \hat{x}_i))$. L represent the generated sequence length, w represent the gound truth sentence and $\hat{x}$ is the predicted sequence embedding from the input $x_t$. $p_{\theta}(w_i | \hat{x}_i)$ follows the softmax distribution. The training objection change to the following function:

$L = \sum_{t=1}^T E_{q(x_t | x_0)} \|\mu_{\theta}(x_t, t) - \mu_{x_t, x_0}\|^2 + -\log(p_{\theta}(w | \hat{x}))$

In our experiments, we found this term significantly influence the model performance{}.

# Our work

# Experiments
Our model is based on Distilbert[] model, Distilbert is a model trained by distiling on BERT[] and has around 40% less parameter than BERT model. 

## fusion: concatenation or elementwise adding 

## relative importance of confidency of prediction and restored feature 
In the ideal case, if model is able of restoring the original text embedding in training step, it should be able to reproduce the label text sequence. However, we found if we remove the term of confidency, the trained model collapse to produce random token to every position of the generated sequence. As a result, we introduced a scalar term that 

## guidance free training

## learning rate

## x_0 prediction or x_{t-n} prediction

## number of x_t predictions


# Conclusion
We present the application of diffusion in image caption task, and proved its validity in limited dataset. Particularly, we identified the certainty term to be an important term in loss function to help model converge, and introduced the adaptive ratio adjustment to balance its importance with other terms. There are various improvements to the model and training process:
- Experiment on output the attention graph of the model, to check the model did focus on the correct region of the image.
- In various cases, the model failed to identify the correct object colour. For example, after correctly identify a girl wearing dress and a shirt, the model mistake the colour of shirt to the colour of the dress. 
- The output text sequence suffers from apparent grammar mistakes, for example, missing subject, repeated words. Additional supervision on the output text grammar might help model reduce such error.
- We trained on raw image data of the dataset. However, image argumentation has proven to be a valid method to improve the performance[]. Performing data argumentation might improve the model's generalizability and help reduce the wrong colour alignment problem as discussed above. 
- Compared to the novelty of image generated by DALL-E2, our model didn't show the advantage brought by diffusion model, which is capable of generating . We suspect the diffusion model was being drift away by the flickr dataset labels, which follows the format of A doing B on place C. By scaling up the training dataset and use more weakly labeled samples, the model might show higher variance in generated text. 
We believe analysing and improving based on the above observations, diffusion as text generation step could achieve comparable or better performance than auto-regressive models. 






ddpm
improved ddpm
ddim
diffusion lm
unet
bert
distilbert
clip
dalle-2
diffusion model beats gan
flickr8k 
flickr30k
COCO
2015 deep caption with multimodal rnn
2016 show attend tell
2016 knowing where to look
2017 cascade recurrent nn
2018 gla
2019 stack vs
// 2019 image caption generation with pos
2019 unsupervised image caption
// 2019 mscap
2019 Variational Autoencoder-Based Multiple ImageCaptioning Using a Caption Attention Map: 基于knowing where to look，生成attention graph，attention graph通过全连接层得到vae mean var，生成文本特征 scale up 传入rnn
2021 CLIP cap
2022 GRIT
// 2022 mplug

2019 Masked Non-Autoregressive Image Captioning decoder使用bert，mask掉所有词逐步生成原文，仅2步即可
2021 Partially Non-Autoregressive Image Captioning
2021 semi Non-Autoregressive Image Captioning 将词组为group，组间auto regressive，组内non autoregressive transformer生成词序