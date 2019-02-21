import tensorflow as tf
import string
from string import digits

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
NUM_LAYERS = 2 # number of layers
LSTM_SIZE = 128 # size of lstm
NUM_CLASSES = 2 #2 classes neg pos
LEARNING_RATE = 0.001
stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
        Apply preprocessing to a single review. You can do anything here that is manipulation
        at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
        RETURN: the preprocessed review in string form.
        """
    # set to lower case
    review_0 = review.lower()
    # word replace
    new_str = str.replace(review_0, '\'re', ' are')
    new_str = str.replace(new_str, 'n\'t', ' not')
    new_str = str.replace(new_str, '\'ve', ' have')
    new_str = str.replace(new_str, '\'m', ' am')
    review_0 = new_str
    #remove the punctuations
    review_1 = review_0.translate(str.maketrans("","",string.punctuation))
    # remove numbers
    review_2 = review_1.translate(str.maketrans("","",digits))
    # remove the stopwords
    words = review_2.split()
    review_3 = [w for w in words if not w in stop_words]
    processed_review = review_3
    
    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats)
    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    
    input_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],name="input_data")
    labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE,NUM_CLASSES],name="labels")
    
    #multiple layers of LSTMs
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(NUM_LAYERS)])
    
    #Initial state of the LSTM memory.
    initial_state = state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)
    
    #dropout
    dropout_keep_prob = tf.placeholder_with_default(input=0.9,shape=[],name="dropout_keep_prob")
    cell_dr = tf.nn.rnn_cell.DropoutWrapper(stacked_lstm, input_keep_prob=1.0, output_keep_prob=dropout_keep_prob)
    
    #get the outputs and the states
    outputs, states = tf.nn.dynamic_rnn(cell_dr,input_data, initial_state=state)

    #Truncated Backpropagation
    #softmax weight and bias
    #softmax_w = tf.Variable(tf.truncated_normal([BATCH_SIZE,NUM_CLASSES]), name="softmax_w")#, stddev=0.1
    #softmax_b = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="softmax_b")
    softmax_w = tf.truncated_normal_initializer(stddev=0.1)
    softmax_b = tf.zeros_initializer()
    

    logits = tf.contrib.layers.fully_connected(outputs[:,-1,:],num_outputs=NUM_CLASSES,activation_fn=tf.nn.softmax,weights_initializer=softmax_w,biases_initializer=softmax_b)
    logits = tf.contrib.layers.dropout(logits, dropout_keep_prob)

    #Calculate mean cross-entropy loss
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    loss = tf.reduce_mean(losses)
    loss = tf.identity(loss,name="loss")
    
    optimizer = tf.train.AdamOptimizer(learning_rate =LEARNING_RATE).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    #optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(loss)

    # Calculate Accuracy
    correct_predictions = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    Accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
    Accuracy = tf.identity(Accuracy, name="accuracy")
    
    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
