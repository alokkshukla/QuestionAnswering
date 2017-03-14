from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import util
from nn_utils import *
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn
import os, sys

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input = None, scope = None, reuse = False):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        length_of_sequence = tf.reduce_sum(masks, 1)
        cell = tf.nn.rnn_cell.LSTMCell(self.size)

        ## Unidirectional RNN 
        # with tf.variable_scope(scope, reuse):
        #     rnn_tensor, state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float64, initial_state = encoder_state_input, sequence_length = length_of_sequence)
        # return rnn_tensor, state

        ## Bi-directional RNN
        with tf.variable_scope(scope, reuse):
            if encoder_state_input == None:
                out, state = bidirectional_dynamic_rnn(cell, cell, inputs, dtype = tf.float64, sequence_length = length_of_sequence)
            else:
                out, state = bidirectional_dynamic_rnn(cell, cell, inputs, initial_state_fw = encoder_state_input[0], initial_state_bw = encoder_state_input[1], dtype = tf.float64, sequence_length = length_of_sequence)
        return out, state

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, question_output, paragraph_output):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        ########## Define weights ###################################
        W_conv1 = weight_variable([5, 5, 4, 16])
        b_conv1 = bias_variable([16])

        W_conv2 = weight_variable([4, 4, 16, 16])
        b_conv2 = bias_variable([16])

        W_conv3 = weight_variable([3, 3, 16, 8])
        b_conv3 = bias_variable([8])

        ############### Define operations #############################
        cnn_input = tf.stack([question_output[0], question_output[1], paragraph_output[0], paragraph_output[1]], axis = 3)

        conv1 = tf.nn.relu(conv2d(cnn_input, W_conv1, 2) + b_conv1)
        pool1 = max_pool_2x2(conv1)
        drop1 = tf.nn.dropout(pool1, 0.8)

        conv2 = tf.nn.relu(conv2d(drop1, W_conv2, 2) + b_conv2)
        pool2 = max_pool_2x2(conv2)
        drop2 = tf.nn.dropout(pool2, 0.7)

        conv3 = tf.nn.relu(conv2d(drop2, W_conv3, 1) + b_conv3)
        pool3 = max_pool_2x2(conv3)
        drop3 = tf.nn.dropout(pool3, 0.6)

        drop3_flat = tf.reshape(drop3, [-1, 1, 1, 1344])

        ## Scores for start
        W_conv_start = weight_variable([1, 1, 1344, 750])
        b_conv_start = bias_variable([750])
        conv_start = conv2d(drop3_flat, W_conv_start, 1) + b_conv_start
        scores_start = tf.reshape(conv_start, [-1, 750])

        ## Scores for end
        W_conv_end = weight_variable([1, 1, 1344, 750])
        b_conv_end = bias_variable([750])
        conv_end = conv2d(drop3_flat, W_conv_end, 1) + b_conv_end
        scores_end = tf.reshape(conv_end, [-1, 750])

        return scores_start, scores_end

class QASystem(object):
    def __init__(self, encoder, decoder, args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        self.encoder = encoder
        self.decoder = decoder

        self.output_size = args.output_size
        self.embed_path = args.embed_path
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate

        self.paragraph_placeholder = tf.placeholder(tf.int32,shape = (None, self.output_size))
        self.mask_paragraph_placeholder = tf.placeholder(tf.int32, shape=(None, self.output_size))

        self.question_placeholder = tf.placeholder(tf.int32, shape = (None, self.output_size))
        self.mask_question_placeholder = tf.placeholder(tf.int32, shape=(None, self.output_size))

        self.answer_start_placeholder = tf.placeholder(tf.int32, shape = (None, self.output_size))
        self.answer_end_placeholder = tf.placeholder(tf.int32, shape = (None, self.output_size))
        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.train_op = get_optimizer(self.optimizer)(self.learning_rate).minimize(self.loss)
        self.merged_summary = tf.summary.merge_all()
        self.train_summary_count = 0
        self.val_summary_count = 0

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        question_output, question_encoded = self.encoder.encode(inputs = self.question_var, masks = self.mask_question_placeholder, scope = "question")
        paragraph_output, paragraph_encoded = self.encoder.encode(inputs = self.paragraph_var, masks = self.mask_paragraph_placeholder, encoder_state_input = question_encoded, scope = "paragraph")
        # self.a_s, self.a_e = self.decoder.decode(question_encoded[1], paragraph_encoded[1])
        self.a_s, self.a_e = self.decoder.decode(question_output, paragraph_output)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("total_loss"):
            l_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.a_s,self.answer_start_placeholder))
            l_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.a_e,self.answer_end_placeholder))
            self.loss = l_1 + l_2
            tf.summary.scalar('loss', self.loss)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings = tf.Variable(np.load(self.embed_path)['glove'])
            paragraph_batch = tf.nn.embedding_lookup(embeddings, self.paragraph_placeholder)
            self.paragraph_var = tf.reshape(paragraph_batch, (-1, self.decoder.output_size, self.encoder.vocab_dim))
            question_batch = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
            self.question_var = tf.reshape(question_batch, (-1, self.decoder.output_size, self.encoder.vocab_dim))


    def optimize(self, session, train_paragraph, train_question, train_ans_s, train_ans_e, mask_paragraph, mask_question):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}
        input_feed[self.answer_start_placeholder] = train_ans_s
        input_feed[self.answer_end_placeholder] = train_ans_e

        input_feed[self.paragraph_placeholder] = train_paragraph
        input_feed[self.question_placeholder] = train_question

        input_feed[self.mask_question_placeholder] = mask_question
        input_feed[self.mask_paragraph_placeholder] = mask_paragraph

        output_feed = [self.merged_summary, self.loss, self.train_op]
        s, loss, _ = session.run(output_feed, input_feed)

        return s, loss


    def test(self, session, valid_paragraph, valid_question, valid_ans_s, valid_ans_e, mask_paragraph, mask_question):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}
        input_feed[self.mask_question_placeholder] = mask_question
        input_feed[self.mask_paragraph_placeholder] = mask_paragraph

        input_feed[self.paragraph_placeholder] = valid_paragraph
        input_feed[self.question_placeholder] = valid_question

        input_feed[self.answer_start_placeholder] = valid_ans_s
        input_feed[self.answer_end_placeholder] = valid_ans_e
        output_feed = [self.loss]
        loss = session.run(output_feed, input_feed)

        return loss

    # def decode(self, session, test_x):
    def decode(self, session, test_paragraph, test_question, mask_paragraph, mask_question):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        input_feed[self.paragraph_placeholder] = test_paragraph
        input_feed[self.mask_paragraph_placeholder] = mask_paragraph

        input_feed[self.question_placeholder] = test_question
        input_feed[self.mask_question_placeholder] = mask_question

        output_feed = [self.a_s, self.a_e]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, p, q, q_mask, p_mask):

        yp, yp2 = self.decode(session, p, q, p_mask, q_mask)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return a_s, a_e

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
            valid_cost = self.test(sess, valid_x, valid_y)

        return valid_cost

    def evaluate_answer(self, session, dataset, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        sample=10
        data_gen = util.load_validate(dataset[0], dataset[1], dataset[2], sample)

        f1_scores = []
        em_scores = []

        ix = 0
        for q, p, ans_s, ans_e, q_mask, p_mask in data_gen:
            ques = np.zeros((1,self.output_size))
            par = np.zeros((1,self.output_size))
            ques_mask = np.zeros((1,self.output_size))
            par_mask = np.zeros((1,self.output_size))
            ans_s = np.zeros((1,self.output_size))
            ans_e = np.zeros((1,self.output_size))
            ques[ix:] = q
            par[ix:,] = p
            par_mask[ix:,] = p_mask
            ques_mask[ix:,] = q_mask
            ix += 1
            a_s, a_e = self.answer(session, par, ques, ques_mask, par_mask)
            a_start = a_s[0]
            a_end = a_e[0]
            if a_end >= a_start:
                pred = p[a_start: a_end]
            else:
                pred = p[a_start: a_start+1]
            ground_truth = p[np.argmax(ans_s): np.argmax(ans_e)]
            pred_str = " ".join(str(e) for e in pred)
            ground_str = " ".join(str(e) for e in ground_truth)

            f1_scores.append(f1_score(pred_str, ground_str))
            em_scores.append(exact_match_score(pred_str, ground_str))

        f1 = sum(f1_scores)/float(len(f1_scores))
        em = sum(em_scores)/float(len(em_scores))

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, saver, dataset_train, dataset_val, train_dir, args, checkpoint_IterNum, train_writer, val_writer):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        batch_size = args.batch_size
        num_epochs = args.epochs
        self.learning_rate = args.learning_rate

        epoch_loss = 0.0
        for epoch_num in range(num_epochs):
            data_gen = util.load_dataset(dataset_train[0], dataset_train[1],  dataset_train[2], batch_size = args.batch_size)
            print ("Epoch " + str(int(checkpoint_IterNum) + epoch_num))
            for ix, i in enumerate(data_gen):
                print ("Batch number: " + str(ix))
                ### Change this to train on entire train data
                if ix == 2:
                    break
                ### 
                ques = np.zeros((batch_size, self.output_size))
                par = np.zeros((batch_size, self.output_size))
                ques_mask = np.zeros((batch_size, self.output_size))
                par_mask = np.zeros((batch_size, self.output_size))
                ans_s = np.zeros((batch_size, self.output_size))
                ans_e = np.zeros((batch_size, self.output_size))
                ix = 0
                for q,p,a_s,a_e,q_mask,p_mask in i:
                    ques[ix:] = q
                    ans_s[ix,:] = a_s
                    ans_e[ix,:] = a_e
                    par[ix:,] = p
                    par_mask[ix:,] = p_mask
                    ques_mask[ix:,] = q_mask
                    ix += 1
                s, loss = self.optimize(session, ques, par, ans_s, ans_e, par_mask, ques_mask)
                train_writer.add_summary(s, self.train_summary_count)
                self.train_summary_count += 1
                epoch_loss = epoch_loss + loss

            print ("Epoch Loss: " + str(epoch_loss))
            print (self.evaluate_answer(session, dataset_val))
            saver.save(session, os.path.join(train_dir, "checkpoint"), global_step = int(checkpoint_IterNum) + epoch_num)
            epoch_loss = 0.0
