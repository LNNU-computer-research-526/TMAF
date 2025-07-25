# Temporal Multi-Scale Attention: Enabling Audio-Visual-Text Multi-Granularity Fusion for Event Localization


## Requirements
### Check the compatibility of your Python packages. If you intend to replicate the results, we suggest using the following configuration, which has been validated through our tests.
```python
Python ==  3.9
Pytorch ==  1.10.2
CUDA ==  11.3
h5py ==  3.13.0
numpy ==  1.23.4
```


## Dataset setup

The AVE dataset can be downloaded from [repo](https://github.com/YapengTian/AVE-ECCV18https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK).

## Training and Evaluating OURS

### Fully-Supervised Setting
The `configs/main.json` contains the main hyper-parameters used for fully-supervised training.

 
```bash
bash video_audio_train_sup.py
```

### Weakly-Supervised Setting
The `configs/weak.json` contains the main hyper-parameters used for weakly-supervised training.


```bash
bash video_audio_train.py
```




## Acknowledgement
We build OURS codebase heavily on the codebase of [YapengTian/AVE-ECCV18](https://github.com/YapengTian/AVE-ECCV18) and [VALOR](https://github.com/Franklin905/VALOR). We sincerely thank the authors for open-sourcing! 
We also thank [CLIP](https://github.com/openai/CLIP) and [CLAP](https://github.com/LAION-AI/CLAP) for open-sourcing pre-trained models.


We would like to thank the authors for releasing their codes. Please also consider citing their works.

