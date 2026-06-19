## How to Run ##
```cd src```

```pip install torch ```

```python main.py```



## My Approach: ##

I choose Paper 3(**Word2Vec**)
So I merged 20 books from Gutenberg for a dataset taking 30,000 words from each book
This led to 11,324 unique words, I trained this model to predict surrounding context of any given words, It learned to map to 50 dimensional vector space

I used skip-gram architecture from the paper and used sliding context window for learning(given in paper)
I just passed center word into model which calculates the CrossEntropyLoss against surrounding context word and backpropogation by using PyTorch
Note that original Paper used some sort of negative sampling approach which makes similar pair true and unrelated pair false,
This was mostly because Paper used 6 billion words dataset with 3 million unique words
so to reduce the time complexity,they used this approach, It also used dynamic window for frequent words whereas i just used fixed window of 2(I have neither time nor computation power to implement that)

Lastly to Prove the accuracy of my model,
Paper method was to use 3 words like
king - man + woman for which the model will give answer as Queen,
I tested it against 100 general 3 word(brother-boy+girl=sister) and 100 book-specific 3 words(darcy-man+woman=elizabeth)
And behold

| NO. OF BOOKS (words per book) | DIMENSION | EPOCH | GENERAL ACCURACY | BOOK-SPECIFIC ACCURACY |
| :--- | :--- | :--- | :--- | :--- |
| 10(30,000) | 20 | 10 | 10/77(12.99%) | 7/71(9.9%) |
| 10(30,000) | 50 | 10 | 8/77(10.3%) | 6/71(8.4%) |
| 20(20,000) | 50 | 10 | 18/84(21.43%) | 11/74(14.86%) |
| 20(30,000) | 50 | 15 | 21/87(24.13%) | 12/94(12.76%) |

(10->15 epoch shows no diff in avg loss)[took 30+ minutes, this is the max my laptop can handle)

Note:21/87 here means 21 words answer was in top 3 of model prediction out of 87 words, rest 13 words were skipped due to words not being in the choosen books)

Now this is obviously disappointing compared to original paper accuracy which was
**Semantic Accuracy:** 67%
**Syntactic Accuracy:** 61%
**Overall Accuracy:** 65%
However using more than 10,000 times lesser words compared to paper, I think its still impressive

**Final Output**

<img width="856" height="420" alt="Screenshot 2026-06-18 214614" src="https://github.com/user-attachments/assets/b8c34912-af45-4791-86ab-6cad5601b6d2" />
