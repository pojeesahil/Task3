from collections import Counter
import os
import re
import urllib
import urllib.request
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

print("Downloading books")
urls={
    "pride_and_prejudice.txt":"https://www.gutenberg.org/files/1342/1342-0.txt",
    "frankenstein.txt":"https://www.gutenberg.org/files/84/84-0.txt",
    "alice_in_wonderland.txt":"https://www.gutenberg.org/files/11/11-0.txt",
    "sherlock_holmes.txt":"https://www.gutenberg.org/files/1661/1661-0.txt",
    "moby_dick.txt":"https://www.gutenberg.org/files/2701/2701-0.txt",
    "tale_of_two_cities.txt":"https://www.gutenberg.org/files/98/98-0.txt",
    "dracula.txt":"https://www.gutenberg.org/files/345/345-0.txt",
    "dorian_gray.txt":"https://www.gutenberg.org/files/174/174-0.txt",
    "great_gatsby.txt":"https://www.gutenberg.org/files/64317/64317-0.txt",
    "grimms_fairy_tales.txt":"https://www.gutenberg.org/files/2591/2591-0.txt"
}
words=[]
for filename,url in urls.items():
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url,filename)
    with open(filename,'r',encoding='utf-8') as f:
        full_text=f.read().lower()

#to remove legal header
    start="start of the project gutenberg ebook"
    if start in full_text:
        story=full_text.split(start)[1]
    else:
        story=full_text
    words.extend(re.sub(r'[^a-z\s]', '', story).split()[:30000])

text=words
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

for i in range(window_size,len(text)-window_size):
    center_word=text[i]
    if center_word in wordint:
        center_idx=wordint[center_word]
        for j in range(i-window_size,i+window_size+1):
            if i!=j and text[j] in wordint:
                centers.append(center_idx)
                contexts.append(wordint[text[j]])
    #print(contexts[:50])
dataset=TensorDataset(torch.tensor(centers,dtype=torch.long),
                      torch.tensor(contexts,dtype=torch.long))
dataloader=DataLoader(dataset,batch_size=1024,shuffle=True)
#print(f"{len(centers)}")
class SimpleWord2Vec(nn.Module):
    def __init__(self,vocabSize,dim):
        super().__init__()
        self.embedding=nn.Embedding(vocabSize,dim)
        self.linear=nn.Linear(dim,vocabSize)
    def forward(self,x):
        embeds=self.embedding(x)
        return self.linear(embeds)

EMBED_DIM=50
EPOCHS=10
model=SimpleWord2Vec(vocabSize,EMBED_DIM)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)

print("training")
for epoch in range(EPOCHS):
    tloss=0.0
    for center_batch,context_batch in dataloader:
        optimizer.zero_grad()
        predictions=model(center_batch)
        loss=criterion(predictions,context_batch)
        loss.backward()
        optimizer.step()
        tloss+=loss.item()
        #print(tloss)
    print(f"Epoch {epoch+1}/{EPOCHS}   = Average Loss: {tloss/len(dataloader):.2f}")
print("\nword test")

def test(w1,w2,w3,model,wordint,intword):
    for w in [w1,w2,w3]:
        if w not in wordint:
            print(f"'{w}' is not in the vocabulary.")
            return
    print(f"\n{w1} - {w2} + {w3} = ?")
    model.eval()
    with torch.no_grad():
        embeddings=model.embedding.weight
        v1=embeddings[wordint[w1]]
        v2=embeddings[wordint[w2]]
        v3=embeddings[wordint[w3]]
        tvector=v1-v2+v3
        similarities=F.cosine_similarity(tvector.unsqueeze(0),embeddings)
        tscores,tindices=torch.topk(similarities,5)
        closest=[(intword[idx.item()],score.item()) for score,idx in zip(tscores,tindices)]
        results=[w for w in closest if w[0] not in [w1,w2,w3]]
        for word,score in results[:3]:
            print(f"{word} (Score: {score})")
test("king","man","woman",model,wordint,intword)
test("brother","boy","girl",model,wordint,intword)
test("bingley","jane","elizabeth",model,wordint,intword)