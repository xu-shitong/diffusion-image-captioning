# diffusion-image-captioning

Research project on image captioning using diffusion language model. Our model is named as CLIP-DiffusionLM.

Best model config is set as hyperparameter in CLIP-DDPM.py file. The model uses configuration of $x_0$-prediction, $\lambda = 0.3$, $lr$ linear decay from 1e-4 to 5e-5, concatenation fusion and non-classification-free guidance. Training time is 5 hours for Flickr8k and 16 hours for Flickr30+8k using AdamW optimizer on a single Nvidia A30 GPU.

## Acknowledgments
We thank Mu Li and Yi Zhu for sharing their insight in various models in vision and NLP field publicly online, Boyang Gu for providing advice in early stage of the research. The computation resource was supported by Imperial College London. 
