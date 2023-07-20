# iCapt - Image Caption Generator
## Overview

This is an Image Caption Generator built using TensorFlow, with the InceptionResNetV2 model as the image encoder and a custom Transformer model as the decoder. The generator is trained on the Flickr8K dataset, enabling it to generate descriptive captions for a wide variety of images.

## Requirements

Make sure you have the following libraries installed:

- TensorFlow
- Keras
- NumPy 
- Pandas 
- Matplotlib 

You can install these dependencies using `pip`:

```bash
pip install tensorflow numpy pandas matplolib 
```

## Dataset

The dataset used for training the Image Caption Generator is the Flickr8K dataset. It contains 8,000 images, each paired with five different captions, making it a total of 40,000 caption-image pairs. The dataset can be downloaded from [link to Flickr8K dataset](https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b).
## Model Architecture

### Image Encoder: InceptionResNetV2

InceptionResNetV2 is used as the image encoder, a powerful convolutional neural network pre-trained on the ImageNet dataset. We extract the image features from the last convolutional layer before feeding them to the decoder.

### Caption Decoder: Custom Transformer

The decoder is a custom Transformer model that takes the image features from the encoder and generates captions sequentially. The Transformer's self and multihead attention mechanism enables the model to focus on relevant parts of the image while generating captions, resulting in more accurate and contextually relevant descriptions.

## How to Use

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/image-caption-generator.git
```

2. Navigate to the project directory:

```bash
cd image-caption-generator
```

3. Download the Flickr8K dataset and place it in the `data/` directory.

4. Train the Image Caption Generator and save the transformer model weights in your directory :

```bash
python icapt.py
```

5. After training, you can use the generator(by using your pretrained transformer) to caption images:

```bash
python generate_caption.py --image path/to/your/image.jpg
```

## Results

The trained Image Caption Generator performs impressively, providing accurate and relevant captions for various images. You can test it on your own images to see how well it works!


---

Feel free to contribute to this project by opening issues or submitting pull requests. If you have any questions or suggestions, please don't hesitate to contact.

**Happy Captioning!**