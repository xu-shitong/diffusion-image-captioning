# Abstract
In this work, we applied denoising diffusion probabilistic models to text generation in image captioning tasks. We show that our CLIP-diffusion-LM *** On the flickr8k dataset, the model showed. By combining samples from flickr8k and flickr30k dataset, our model showed ... performance. In addition, the model achieved ... zero shot performance in COCO 2015 image caption task. Our code is available at ...

# Introduction
(image captioning papers from paper with code)


In recent years, CLIP encoders are widely as feature extractors for various tasks[]. Most recently, the DALL-E 2 model has shown significant performance in text-to-image generation tasks. The application of diffusion model in generation step allows higher variation and novelty in generated images. 





Our work experimented in reverse direction of DALL-E 2, aiming at generating text with high novelty from images. To the best of our knowledge, there has not been previous work on applying diffusion model on text generation in image captioning tasks. 

denoising diffusion probabilistic model has seen significant advance in generation tasks. 


# Related work

## Diffusion models
In recent years, denoising diffusion probabilistic model has seen significant advance in generation tasks. [] proposed the DDPM model to simplified the loss function by only letting models to predict the noise in generation steps, and proposed alternative loss functions by removing the weight terms. Their work showed diffusion models' capability of generating high resolution samples. Based on DDPM, improved ddpm [] which proposed several improvments based on DDPM, including setting variance to be learnable parameters, apply cosine instead of linear noise schedule, and speed up forward process by reducing forward steps. 

Diffusion model beat GAN[] is another work which used larger model, more attention heads, multiscaled attention and classifier guidance(?is the classifier guidance reduce generation step or higher score) for reducing generation steps. Classifuer guided diffusion is a technique of using a classifier to help generating features with higher . The classifier is pretrained on dataset with noise added. To guide the generation, the model's output is feed into the classifier, to get gradient on which direction to optimise parameter in order to generate image closer to target class.

To avoid training classifier for guiding model, classifier free guidance technique is proposed, the task is to provide generator both guided or unguided samples, and train an additional model to predict the relationship in 2 models' output. Example of applying classifier free guidance includes GLIDE[], (other class free guide work?)

Diffusion lm[] which is a recent work on applying continuous diffusion model on text generation, this paper provides various techniques to improving the performance of continuous diffusion model on text generation. 

ddim[] reduced the variance in forward process. The result showed that by reducing variance to 0, the deterministic model achieved higher FID score on (? dataset). 

In these work, U-net[] is commonly used in each forward process since mose previous diffusion modes were based on images. Diffusion-lm used BERT[] model in each generation step. In our work, due to the limit of training ability, Distil-bert[] is used instead. Distilbert is a model trained by distiling on BERT (large small ...?) and has around 40% less parameter than BERT (large, small...?).


## CLIP feature extraction
CLIP is an image-text feature extracting model trained using contrastive learning. It used 40M image-text sample pairs from the internet. The model has achieved comparable down-stream task zero shot performances in multiple datasets. In our implementation, the CLIP image features are used as guidance for models to generate text sequence. From our experiments, using CLIP feature as an additional sequence token performs better than adding it as time embedding.{} 

Pretrained CLIP[] model is used for feature extraction in our model. CLIP is a feature extraction model trained using contrastive learning on WebImageText dataset, which has over 40m samples. The model achieved significant zero shot performance on various datasets. Some performaces even better than previous sota models trained directly on the dataset. 

Dall-e 2[] is a recent work on using CLIP and diffusion model for image generation task. The model used CLIP model for extracting feature from text, predict the corrisponding image CLIP feature through prior network, then use predicted image CLIP feature for final image generation. The model achieved significant novelty in generated images. 

# Background

## Diffusion models
The training of denoise diffusion probabilistic model involves generation of noised samples (forward process), and denoising based on model's output (backward process). Let x_0 be the original feature, the forward process generates a sequence of T noised features $[x_1, ... x_T]$. Each x_t at step t is generated from prefious step feature x_{t-1} by probability distribution $p(x_t | x_{t-1}) = N(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$. From reparameterization trick, the x_t at any step t could be directly generate from x_0: $x_0 = ...$. The training objective is to minimize the negative log-likelihood of generating x_0 from arbitary x_t: that is to minimize $...$. By [] the loss could be split to multiple terms: $...$. By assuming the reverse variance remain constant, L_T term is constant and ignored in loss calculation step. Lt terms corrisponds to the reverse step losses when t > 1. To compute the loss in each reverse step, L1 [] or MSE losses [] were commonly used 

Due to the large generation step number (1000 as proposed in []), and the generation step being autoregressive, the reverse diffusion is significantly slower than the other generative models ... for gan and ... for diffusion). Multiple stetegies were proposed to accelerate the generation process. In Improved DDPM [] a subset of generation steps is selected. Model is trained to predict ... . In diffusion-lm the model is trained directly predict the x_0 instead of the intermediate steps containing noise. In our experiments, the ... showed better reproduce quality compared with... {}. (In addition, autoregressively apply the output of x_0 trained model further improved the performance. {}

# Our work

# Experiments
Our model is based on Distilbert model, 

# Conclusion





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