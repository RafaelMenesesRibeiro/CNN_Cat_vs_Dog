# CNN_Cat_vs_Dog

Convolutional Neural Network to distinguish between cats and dogs given only an image.  
Uses Tensorflow library for the CNN model.

![screenshot](https://raw.githubusercontent.com/RafaelRibeiro97/CNN_Cat_vs_Dog/master/media/Capture.PNG)

## Requirements
* Kaggle Dataset - [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) - only <b>train.zip</b>
* Tensorflow / Tensorflow-gpu
  - CUDA Toolkit 8.0 (optional - to use with the gpu version)
  - cuDNN 7 (optional - to use with the gpu version)
* TFlearn
* OpenCV
* Numpy
* TQDM (optional - no added functionality or better performance - only for pretty progress bars)

## How To Use
```python
#Set TRAIN_DIR in CNN_catdog_train.py:17 to the directory where the downloaded images are.
TRAIN_DIR = ''
```

```bash
#Run CNN_catdog_train.py to train/load the model. Input an image path (to test) when asked.
$ python CNN_catdog_train.py
```


