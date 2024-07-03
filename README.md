# iCapt

## Overview

This is an Image Caption Generator built using TensorFlow, with the InceptionResNetV2 model as the image encoder and a custom Transformer model as the decoder. The generator is trained on the Flickr8K dataset, enabling it to generate descriptive captions for a wide variety of images.

The dataset used for training the Image Caption Generator is the Flickr8K dataset. It contains 8,000 images, each paired with five different captions, making it a total of 40,000 caption-image pairs. The dataset can be downloaded from [link to Flickr8K dataset](https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b).
## Model Architecture

### Image Encoder: InceptionResNetV2

InceptionResNetV2 is used as the image encoder, a powerful convolutional neural network pre-trained on the ImageNet dataset. We extract the image features from the last convolutional layer before feeding them to the decoder.

### Caption Decoder: Custom Transformer

The decoder is a custom Transformer model that takes the image features from the encoder and generates captions sequentially. The Transformer's self and multihead attention mechanism enables the model to focus on relevant parts of the image while generating captions, resulting in more accurate and contextually relevant descriptions.

## How to Use

After cloning this repository to your local machine using : 

```bash
git clone https://github.com/your-username/image-caption-generator.git
```
follow the below steps - 

### 1. Develop and save the model running `python3 /model/train.py`


### 2. Create Docker container

```bash
docker build -t app-name .

docker run -p 80:80 app-name
```

### 3. Go to `0.0.0.0/docs` to upload your image and get the caption!