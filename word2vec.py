#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/word2vec.ipynb

import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.

# Set the number of negative samples per positive context.
# Key point: num_ns (the number of negative samples per a positive context word) in the [5, 20] range
# is shown to work best for smaller datasets, while num_ns in the [2, 5] range suffices for larger datasets.
num_ns = 4

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
# Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

# Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

# Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels




# Load the TensorBoard notebook extension
# %load_ext tensorboard
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

path_to_file = tf.keras.utils.get_file('corpus_50.txt', "file:\\C:/Users/hello/Desktop/AES_WE/corpus_50.txt")
# path_to_file = tf.keras.utils.get_file('shakespeare.txt', "file:\\C:/Users/hello/Desktop/AES_WE/shakespeare.txt")
# Read the text from the file and print the first few lines:
with open(path_to_file, encoding="utf8") as f:
  lines = f.read().splitlines()
for line in lines[:20]:
  print(line)

# Use the non empty lines to construct a tf.data.TextLineDataset object for the next steps:
text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

# Now, create a custom standardization function to lowercase the text and
# remove punctuation.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')


# Define the vocabulary size and the number of words in a sequence.
vocab_size = 4096
sequence_length = 10

# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the
# same length.
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Call TextVectorization.adapt on the text dataset to create vocabulary.
# Once the state of the layer has been adapted to represent the text corpus, the vocabulary can be accessed
# with TextVectorization.get_vocabulary.
# This function returns a list of all vocabulary tokens sorted (descending) by their frequency.
vectorize_layer.adapt(text_ds.batch(1024))

# Save the created vocabulary for reference.
inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])

# The vectorize_layer can now be used to generate vectors for each element in the text_ds (a tf.data.Dataset).
# Apply Dataset.batch, Dataset.prefetch, Dataset.map, and Dataset.unbatch.

# Vectorize the data in text_ds.
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

# You now have a tf.data.Dataset of integer encoded sentences. To prepare the dataset for training a word2vec model,
# flatten the dataset into a list of sentence vector sequences.
# This step is required as you would iterate over each sentence in the dataset to produce positive and negative examples.
# Note: Since the generate_training_data() defined earlier uses non-TensorFlow Python/NumPy functions,
# you could also use a tf.py_function or tf.numpy_function with tf.data.Dataset.map.

sequences = list(text_vector_ds.as_numpy_iterator())
print(len(sequences))

# Inspect a few examples from sequences:

for seq in sequences[:5]:
  print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

# Generate training examples from sequences
# sequences is now a list of int encoded sentences.
# Just call the generate_training_data function defined earlier to generate training examples for the word2vec model.
# To recap, the function iterates over each word from each sequence to collect positive and negative context words.
# Length of target, contexts and labels should be the same, representing the total number of training examples.

targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED)

targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")

# Configure the dataset for performance
# To perform efficient batching for the potentially large number of training examples,
# use the tf.data.Dataset API. After this step, you would have a tf.data.Dataset object of (target_word, context_word),
# (label) elements to train your word2vec model!

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

# Apply Dataset.cache and Dataset.prefetch to improve performance:

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)

# Subclassed word2vec model
# Use the Keras Subclassing API to define your word2vec model with the following layers:
#
# target_embedding: A tf.keras.layers.Embedding layer, which looks up the embedding of a word when it appears as a target word. The number of parameters in this layer are (vocab_size * embedding_dim).
# context_embedding: Another tf.keras.layers.Embedding layer, which looks up the embedding of a word when it appears as a context word. The number of parameters in this layer are the same as those in target_embedding, i.e. (vocab_size * embedding_dim).
# dots: A tf.keras.layers.Dot layer that computes the dot product of target and context embeddings from a training pair.
# flatten: A tf.keras.layers.Flatten layer to flatten the results of dots layer into logits.
# With the subclassed model, you can define the call() function that accepts (target, context) pairs which can then be passed into their corresponding embedding layer. Reshape the context_embedding to perform a dot product with target_embedding and return the flattened result.
#
# Key point: The target_embedding and context_embedding layers can be shared as well. You could also use a concatenation of both embeddings as the final word2vec embedding.



class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots


# Define loss function and compile model
# For simplicity, you can use tf.keras.losses.CategoricalCrossEntropy as an alternative to the negative sampling loss.
# If you would like to write your own custom loss function, you can also do so as follows:

def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

# It's time to build your model!
# Instantiate your word2vec class with an embedding dimension of 128 (you could experiment with different values).
# Compile the model with the tf.keras.optimizers.Adam optimizer.


embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
word2vec.fit(dataset, epochs=40)

# Embedding lookup and analysis
# Obtain the weights from the model using Model.get_layer and Layer.get_weights.
# The TextVectorization.get_vocabulary function provides the vocabulary
# to build a metadata file with one token per line.

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

# Create and save the vectors and metadata files:
out_v = io.open("C:/Users/hello/Desktop/AES_WE/vectors.tsv", 'w', encoding='utf-8')
out_m = io.open("C:/Users/hello/Desktop/AES_WE/metadata.tsv", 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()





