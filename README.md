# Text Classification Using Self Attention

An implementation of the paper : A Structured Self-attentive Sentence Embedding ( https://arxiv.org/pdf/1703.03130.pdf ) using TensorFlow

For this bit of code, the problem is assumed to be a simple binary classification problem ( although my data set had class imabalance, which is also addressed in the code ), and it can be changed to multiclass by simply replacing the the sigmoid in the loss function, with a softmax, and removing the class weights
