
# coding: utf-8

# # Text Generation - Charles Dickens
# 
# In this notebook, I'll build a character-wise RNN trained on five books from Charles Dickens: 'A Tale of Two Cities,' 'Great Expectations,' 'Oliver Twist,' 'David Copperfield,' and 'A Christmas Carol.' These books were chosen because they were available on [www.gutenberg.org](http://www.gutenberg.org), they all come from the same, famous author, and together they add up to a significant sum of text, 4.7 MB. Although any one of these books could suffice for this simple project, by having more text it should help the model to avoid overfitting. 
# 
# Much of this code is from a lesson on recurrent neural networks, as part of my Deep Learning Nanodegree Foundation from Udacity. Here is the lesson on [Github](https://github.com/udacity/deep-learning/tree/master/intro-to-rnns).

# In[1]:

import time
from collections import namedtuple
import numpy as np
import tensorflow as tf


# First we'll load the text file and convert it into integers for our network to use. Here I'm creating a couple dictionaries to convert the characters to and from integers. Encoding the characters as integers makes it easier to use as input in the network.

# In[2]:

with open('Dickens_Books.txt', 'r') as f:
    text=f.read()
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
chars = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


# Let's check out the first sentence of our text, which is from Great Expectations.

# In[3]:

text[:218]


# And we can see the characters encoded as integers.

# In[4]:

chars[:218]


# ## Making training and validation batches
# 
# Now I need to split up the data into batches, and into training and validation sets. I should be making a test set here, but I'm not going to worry about that. My test will be if the network can generate new text.
# 
# Here I'll make both input and target arrays. The targets are the same as the inputs, except shifted one character over. I'll also drop the last bit of data so that I'll only have completely full batches.
# 
# The idea here is to make a 2D matrix where the number of rows is equal to the batch size. Each row will be one long concatenated string from the character data. We'll split this data into a training set and validation set using the `split_frac` keyword. This will keep 90% of the batches in the training set, the other 10% in the validation set.

# In[5]:

def split_data(chars, batch_size, num_steps, split_frac=0.9):
    """ 
    Split character data into training and validation sets, inputs and targets for each set.
    
    Arguments
    ---------
    chars: character array
    batch_size: Size of examples in each of batch
    num_steps: Number of sequence steps to keep in the input and pass to the network
    split_frac: Fraction of batches to keep in the training set
    
    
    Returns train_x, train_y, val_x, val_y
    """
    
    slice_size = batch_size * num_steps
    n_batches = int(len(chars) / slice_size)
    
    # Drop the last few characters to make only full batches
    x = chars[: n_batches*slice_size]
    y = chars[1: n_batches*slice_size + 1]
    
    # Split the data into batch_size slices, then stack them into a 2D matrix 
    x = np.stack(np.split(x, batch_size))
    y = np.stack(np.split(y, batch_size))
    
    # Now x and y are arrays with dimensions batch_size x n_batches*num_steps
    
    # Split into training and validation sets, keep the first split_frac batches for training
    split_idx = int(n_batches*split_frac)
    train_x, train_y= x[:, :split_idx*num_steps], y[:, :split_idx*num_steps]
    val_x, val_y = x[:, split_idx*num_steps:], y[:, split_idx*num_steps:]
    
    return train_x, train_y, val_x, val_y


# Now I'll make my data sets and we can check out what's going on here. Here I'm going to use a batch size of 10 and 50 sequence steps.

# In[6]:

train_x, train_y, val_x, val_y = split_data(chars, 10, 50)


# In[7]:

train_x.shape


# Looking at the size of this array, we see that we have rows equal to the batch size. When we want to get a batch out of here, we can grab a subset of this array that contains all the rows but has a width equal to the number of steps in the sequence. The first batch looks like this:

# In[8]:

train_x[:,:50]


# I'll write another function to grab batches out of the arrays made by `split_data`. Here each batch will be a sliding window on these arrays with size `batch_size X num_steps`. For example, if we want our network to train on a sequence of 100 characters, `num_steps = 100`. For the next batch, we'll shift this window the next sequence of `num_steps` characters. In this way we can feed batches to the network and the cell states will continue through on each batch.

# In[9]:

def get_batch(arrs, num_steps):
    batch_size, slice_size = arrs[0].shape
    
    n_batches = int(slice_size/num_steps)
    for b in range(n_batches):
        yield [x[:, b*num_steps: (b+1)*num_steps] for x in arrs]


# ## Building the model
# 
# Below is a function where I build the graph for the network.

# In[10]:

def build_rnn(num_classes, batch_size=50, num_steps=50, hidden_size=128, num_layers=2,
              learning_rate=0.001, grad_clip=5, sampling=False):
    
    # When we're using this network for sampling later, we'll be passing in
    # one character at a time, so providing an option for that
    if sampling == True:
        batch_size, num_steps = 1, 1

    tf.reset_default_graph()
    
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    
    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    # One-hot encoding the input and target characters
    x_one_hot = tf.one_hot(inputs, num_classes)
    y_one_hot = tf.one_hot(targets, num_classes)

    ### Build the RNN layers
    # Use a basic GRU cell
    gru = tf.contrib.rnn.GRUCell(hidden_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(gru, output_keep_prob=keep_prob)
    
    # Stack up multiple GRU layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

    ### Run the data through the RNN layers
    # This makes a list where each element is on step in the sequence
    rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(x_one_hot, num_steps, 1)]
    
    # Run each sequence step through the RNN and collect the outputs
    outputs, state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=initial_state)
    final_state = state
    
    # Reshape output so it's a bunch of rows, one output row for each step for each batch
    seq_output = tf.concat(outputs, axis=1)
    output = tf.reshape(seq_output, [-1, hidden_size])
    
    # Now connect the RNN putputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((hidden_size, num_classes), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(num_classes))
    
    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and batch
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    # Use softmax to get the probabilities for predicted characters
    preds = tf.nn.softmax(logits, name='predictions')
    
    # Reshape the targets to match the logits
    y_reshaped = tf.reshape(y_one_hot, [-1, num_classes])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    cost = tf.reduce_mean(loss)

    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    # Export the nodes
    export_nodes = ['inputs', 'targets', 'initial_state', 'final_state',
                    'keep_prob', 'cost', 'preds', 'optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph


# ## Hyperparameters
# 
# Here I'm defining the hyperparameters for the network. 
# 
# * `batch_size` - Number of sequences running through the network in one pass.
# * `num_steps` - Number of characters in the sequence the network is trained on. Larger is better typically, the network will learn more long range dependencies. But it takes longer to train. 100 is typically a good number here.
# * `lstm_size` - The number of units in the hidden layers.
# * `num_layers` - Number of hidden LSTM layers to use
# * `learning_rate` - Learning rate for training
# * `keep_prob` - The dropout keep probability when training. If you're network is overfitting, try decreasing this.
# 
# Here's some good advice from [Andrej Karpathy](https://github.com/karpathy/char-rnn#tips-and-tricks) on training the network.

# In[11]:

batch_size = 50
num_steps = 150
hidden_size = 800
num_layers = 2
learning_rate = 0.0005
keep_prob = 0.75


# ## Training
# 
# Time for training which is pretty straightforward. Here I pass in some data, and get an LSTM state back. Then I pass that state back in to the network so the next batch can continue the state from the previous batch. And every so often (set by `save_every_n`) I calculate the validation loss and save a checkpoint.
# 
# Here I'm saving checkpoints with the format
# 
# `i{iteration number}_l{# hidden layer units}_v{validation loss}.ckpt`

# In[12]:

epochs = 6
# Save every N iterations
save_every_n = 200
train_x, train_y, val_x, val_y = split_data(chars, batch_size, num_steps)

model = build_rnn(len(vocab), 
                  batch_size=batch_size,
                  num_steps=num_steps,
                  learning_rate=learning_rate,
                  hidden_size=hidden_size,
                  num_layers=num_layers)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Use the line below to load a checkpoint and resume training
    #saver.restore(sess, 'checkpoints/______.ckpt')
    
    n_batches = int(train_x.shape[1]/num_steps)
    iterations = n_batches * epochs
    for e in range(epochs):
        
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for b, (x, y) in enumerate(get_batch([train_x, train_y], num_steps), 1):
            iteration = e*n_batches + b
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.cost, model.final_state, model.optimizer], 
                                                 feed_dict=feed)
            loss += batch_loss
            end = time.time()
            print('Epoch {}/{} '.format(e+1, epochs),
                  'Iteration {}/{}'.format(iteration, iterations),
                  'Training loss: {:.4f}'.format(loss/b),
                  '{:.4f} sec/batch'.format((end-start)))
        
            
            if (iteration%save_every_n == 0) or (iteration == iterations):
                # Check performance, notice dropout has been set to 1
                val_loss = []
                new_state = sess.run(model.initial_state)
                for x, y in get_batch([val_x, val_y], num_steps):
                    feed = {model.inputs: x,
                            model.targets: y,
                            model.keep_prob: 1.,
                            model.initial_state: new_state}
                    batch_loss, new_state = sess.run([model.cost, model.final_state], feed_dict=feed)
                    val_loss.append(batch_loss)

                print('Validation loss:', np.mean(val_loss),
                      'Saving checkpoint!')
                saver.save(sess, "Charles_Dickens_i{}_l{}_v{:.3f}.ckpt".format(iteration, hidden_size, np.mean(val_loss)))


# #### Saved checkpoints
# 
# Read up on saving and loading checkpoints here: https://www.tensorflow.org/programmers_guide/variables

# In[13]:

tf.train.get_checkpoint_state('checkpoint')


# ## Sampling
# 
# Now that the network is trained, we'll can use it to generate new text. The idea is that we pass in a character, then the network will predict the next character. We can use the new one, to predict the next one. And we keep doing this to generate all new text. I also included some functionality to prime the network with some text by passing in a string and building up a state from that.
# 
# The network gives us predictions for each character. To reduce noise and make things a little less random, I'm going to only choose a new character from the top N most likely characters.
# 
# 

# In[14]:

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


# In[15]:

def sample(checkpoint, n_samples, hidden_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    model = build_rnn(vocab_size, hidden_size=hidden_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)


# Here, pass in the path to a checkpoint and sample from the network.

# ### 200 iterations, loss = 2.22

# In[21]:

checkpoint = "/Users/Dave/Desktop/Programming/Personal Projects/text_generation_great_expectations/great_expectations_i200_l800_v2.220.ckpt"
samp = sample(checkpoint, 1000, hidden_size, len(vocab), prime="she looked ")
print(samp)


# ### 1600 iterations, less = 1.32

# In[22]:

checkpoint = "/Users/Dave/Desktop/Programming/Personal Projects/text_generation_great_expectations/great_expectations_i1600_l800_v1.320.ckpt"
samp = sample(checkpoint, 1000, hidden_size, len(vocab), prime="she looked ")
print(samp)


# ### 3400 iterations, loss = 1.18 (final iteration)

# In[20]:

checkpoint = "/Users/Dave/Desktop/Programming/Personal Projects/text_generation_great_expectations/great_expectations_i3400_l800_v1.184.ckpt"
samp = sample(checkpoint, 1000, hidden_size, len(vocab), prime="she looked ")
print(samp)


# ## Summary

# Given the length of time that it takes to train this network on my laptop, ~9 hours, more experimenting could be done to improve results, but I would rather move on to other projects. I have some thoughts about how to improve results:
# - Increase the num_steps so that each character gains a greater understanding of its relation to the other characters.
# - Increase the hidden_size. This should help the network to learn better (but it may also fill up your RAM as it did to me).
# - Use a custom made cell to replace the GRU cell that I used. An example can be found [here](http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html).
# 
# Although I did not make significant changes to the code, my choice of input and parameters helped to improve the results. By using more text, overfitting could be reduced and the size of the testing data could be increased. Increasing the num_steps, hidden_size, and dropout, as well as decreasing the learning rate all helped to improve the performance of the model.
# 
# Looking at the output, we can really see how much the model has learned. Spelling mistakes are minimal, punctuation is very good, as well as the structure of a paragraph. 

# In[ ]:



