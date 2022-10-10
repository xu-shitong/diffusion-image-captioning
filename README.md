# diffusion-image-captioning

Research project on image captioning using diffusion language model. Our model is named as CLIP-DiffusionLM.

We provide the extracted CLIP feature for Flickr8k dataset in repo https://github.com/xu-shitong/flickr8k-CLIP-freature} and can be downloaded as shown in CLIP-DDPM.ipynb file. However, due to file size limit, we do not disclose extracted CLIP feature for Flickr30k dataset. User will need to extract their own.

Best model hyperparameter config and training code is in CLIP-DDPM.py file. The model uses configuration of $x_0$-prediction, $\lambda = 0.3$, $lr$ linear decay from 1e-4 to 5e-5, concatenation fusion and non-classification-free guidance. Training time is 5 hours for Flickr8k and 16 hours for Flickr30+8k using AdamW optimizer on a single Nvidia A30 GPU.

## Acknowledgments
We thank Mu Li and Yi Zhu for sharing their insight in various models in vision and NLP field publicly online, Boyang Gu for providing advice in early stage of the research. The computation resource was supported by Imperial College London. 
