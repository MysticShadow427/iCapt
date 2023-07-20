import argparse
from tensorflow.keras import load_model
from icapt import evaluate

parser = argparse.ArgumentParser(description='Image Caption Generator')
parser.add_argument('--path', type=str, required=True, help='Path to the image for caption generation')
args = parser.parse_args()



def generate_caption(image_path):
    # loading the pretrained model weights
    model = load_model('path/to/your/model/')
    # generating the caption by importing the function
    evaluate(args)

if __name__ == "__main__":
    generate_caption(args.path)
