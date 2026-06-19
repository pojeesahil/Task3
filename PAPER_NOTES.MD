Central claim:simple basic models like skip-gram works better than complex model with hidden layers,better way is to train the simple model on large dataset

Core algorithm:We need to make simple model consisting of just input map and output 
(linear layer),model can learn by center words against context words and learning by backpropogation and we need to test its accuracy

Dataset:Paper model dataset was Google news ,it trained on 6 billion words out of which 3 million were unique, Accuracy was evaluated by vector eqn like “king” - “man” + “woman” = “Queen” ,if model answers it correctly,It was marked as correct, The results were compared against previous language models
