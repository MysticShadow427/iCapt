from evaluate import evaluate
from load_data import get_tokenizer

__version__ = "0.1.0"
tokenizer = get_tokenizer()

def predict_pipeline(image):
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']
    caption,result,attention_weights = evaluate(image)

    #remove "<unk>" in result
    for i in caption:
        if i=="<unk>":
            caption.remove(i)
    
    predicted_caption = ' '.join(caption)
    return predicted_caption