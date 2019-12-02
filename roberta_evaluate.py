import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa  # load the Arc QA task
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'data/CommonsenseQA')
#roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
roberta.eval()  # disable dropout
roberta.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0

def getBaseAsciiValue(label):
    asciiValue = ord(label)
    if(asciiValue >= 49 and asciiValue <= 57):
        return 49
    elif(asciiValue >= 65 and asciiValue <= 90):
        return 65
    elif(asciiValue >= 97 and asciiValue <= 122):
        return 97
    return -1
            
with open('data/CommonsenseQA/ARC-Easy-Test_var4.jsonl') as h:
    for line in h:
        example = json.loads(line)
        scores = []
        for choice in example['question']['choices']:
            input = roberta.encode(
                'Q: ' + example['question']['stem'],
                'A: ' + choice['text'],
                no_separator=True
            )
            score = roberta.predict('sentence_classification_head', input, return_logits=True)
            scores.append(score)
        pred = torch.cat(scores).argmax()
        answer = ord(example['answerKey']) - getBaseAsciiValue(example['answerKey'])
        nsamples += 1
        if pred == answer:
            ncorrect += 1
        print(str(pred) + "\t" + str(answer) + "\t" + str(ncorrect))

print(nsamples)        
print('Accuracy: ' + str(ncorrect / float(nsamples)))
# Accuracy: 0.7846027846027847
