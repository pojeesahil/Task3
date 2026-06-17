from collections import Counter
import re
import urllib
from torch.utils.data import TensorDataset, DataLoader
import torch

print("Downloading book[Pride and Prejudice]")
url="https://www.gutenberg.org/files/1342/1342-0.txt"
response=urllib.request.urlopen(url)
full_text=response.read().decode('utf-8').lower()

#to remove legal header
start="start of the project gutenberg ebook"
if start in full_text:
    story=full_text.split(start)[1]
else:
    story=full_text

words=re.sub(r'[^a-z\s]','',story).split()
text=words[:100000]
#print(f"{len(text)} words.")

word_counts=Counter(text)
vocab=[]
for w,count in word_counts.items():
    if count>=3:
        vocab.append(w)
vocabSize=len(vocab)
wordint={}
intword={}
for i in range(len(vocab)):
    current_word=vocab[i]
    wordint[current_word]=i
    intword[i]=current_word

print(f"{vocabSize} unique words.")
centers=[]
contexts=[]
window_size=2

for i in range(window_size,len(text)-window_size+1):
    center_word=text[i]
    if center_word in wordint:
        center_idx=wordint[center_word]
        for j in range(i-window_size,i+window_size):
            if i!=j and text[j] in wordint:
                centers.append(center_idx)
                contexts.append(wordint[text[j]])

dataset=TensorDataset(torch.tensor(centers,dtype=torch.long),
                      torch.tensor(contexts,dtype=torch.long))
dataloader=DataLoader(dataset,batch_size=1024,shuffle=True)
#print(f"{len(centers)}")