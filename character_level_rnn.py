import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# 1. Prepare text data
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding = 'utf-8') # retrieved text

# Tokenize the text and create sequences
vocab = sorted(set(text)) # set of unique characters in the text
ids_from_chars = tf.keras.layers.StringLookup(vocabulary = list(vocab), # tokenizing function
                                              mask_token = None)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary =  list(vocab), # detokenizing function to use when generating text
                                              invert = True,
                                              mask_token = None)
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8')) # tokens
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids) # stream of tokens
SEQ_LENGTH = 100
sequences = ids_dataset.batch(SEQ_LENGTH+1, drop_remainder = True) # sequences

# Create (input, label) pairs and group into batches
def split_input_target(sequence): # function to return an (input sequence, label sequence) pair
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target) # dataset containing (input, label) pairs

BATCH_SIZE = 64
BUFFER_SIZE = 10000 # number of data points to read at a time
dataset = (dataset
           .shuffle(BUFFER_SIZE) # shuffle data to reduce bias caused by learning data in order
           .batch(BATCH_SIZE, drop_remainder=True) # group data into batches for faster processing
           .prefetch(tf.data.experimental.AUTOTUNE)) # prefetch next batch to reduce delay between training steps

# 2. Build LSTM model
vocab_size = len(vocab) # number of unique characters in the text
embedding_dim = 256 # vector length of embedding layer's output
lstm_units = 512 # vector length of LSTM layers' outputs

model = Sequential([
  Embedding (vocab_size, # embedding layer to extract features
             embedding_dim #,
            # input_length = SEQ_LENGTH
             ),
  LSTM (lstm_units, # long short-term memory (impl. of RNN) layer to include prediction history in input
        return_sequences = True),
  LSTM (lstm_units,
        return_sequences = True), # added to layer: omitting resulted in loss of SEQ_LENGTH dimension in output shape
  Dense (vocab_size, # dense layer to produce prediction
         activation = 'softmax')
])

# 3. Train model (set computation hardware to GPU)
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam')
NUM_EPOCHS = 5
model.fit(dataset,
          epochs = NUM_EPOCHS)

# # 4. Generate text
# def generate_text(seed_text, length = 100, temperature = 1.0):
#   SEED_LENGTH = len(seed_text)

#   input_chars = tf.strings.unicode_split(seed_text, 'UTF-8')
#   input_ids = tf.expand_dims(ids_from_chars(input_chars), axis = 0) # input ids in (batch = 1, sample) format
#   generated_chars = []

#   for _ in range(length):
#     # Predict the next id given the input
#     predictions = model(input_ids) # probability distributions of predictions at each layer of the model
#     predictions = predictions[:, -1, :] # probability distribution at last char in seed_text
#     predictions = predictions / temperature # a higher temperature evens out probabilities across predictions, increasing variability
#     predicted_id_tensor = tf.random.categorical(predictions, num_samples = 1) # tensor of id randomly sampled from predictions

#     # Add the predicted char to the list of generated chars
#     predicted_char = chars_from_ids(predicted_id_tensor).numpy().item().decode('utf-8') # char mapped to predicted id and converted from tensor
#     generated_chars.append(predicted_char)

#     # Add the predicted id to the list of input ids
#     input_ids = tf.concat([input_ids, predicted_id_tensor], axis = 1)
#     input_ids = input_ids[:, -SEED_LENGTH:] # input_ids restricted to length of seed text

#   return seed_text + ''.join(generated_chars)

# Define class that runs the model over one char and returns the predicted char and the model's state
class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature = 1.0):
    super().__init__()
    self.temperature = temperature # temperature is used to amplify (> 1) / flatten (< 1) the distribution curve,
                                   # increasing/decreasing prediction variability
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to suppress "[UNK]" token generation ([UNK] is used to map chars in input that are not in the training vocab)
    skip_id = self.ids_from_chars(['[UNK]'])[:, None] # 1x1 tensor of the "[UNK]" token; "[:, None]" transforms the tensor to 1x1
    unk_mask = tf.SparseTensor( # sparse tensor of mask (for ease of construction)
        values=[-float('inf')]*len(skip_id), # value to place at index = [UNK] ID
        indices=skip_id, # indices where value should be placed
        dense_shape=[len(ids_from_chars.get_vocabulary())] # shape of tensor (i.e. vocab size) if expanded to dense shape
    )
    self.prediction_mask = tf.sparse.to_dense(unk_mask) # dense tensor of unk_mask (for actual implementation)

  @tf.function # annotation that saves the Python function as its interpreted TensorFlow graph; increases efficiency
  def generate_one_step(self, inputs, states = None): # function to perform one step
    # Convert input to tensor of IDs
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Predict the next ID
    predictions, states = self.model(inputs = input_ids, # probability distributions and model states
                                      states = states,    # of predictions at each layer of the model
                                      return_state = True)
    predictions = predictions[:, -1, :] # probability distribution at last char in input
    predictions = predictions / self.temperature # amplified/flattened distribution curve
    predictions = predictions + self.prediction_mask # predictions with [UNK] suppressed by setting its probability to -infinity
    predicted_id_tensor = tf.random.categorical(predictions, num_samples = 1) # tensor of ID randomly sampled from predictions
    predicted_id_tensor = tf.squeeze(predicted_id_tensor, axis=-1) # tensor with vocab_size dimension removed

    # Convert the predicted ID to its char
    predicted_chars = self.chars_from_ids(predicted_id_tensor)

    # Return the predicted char and the model state
    return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

# 4. Generate text
def generate_text(seed_text, length = 100, temperature = 1.0):
  states = None
  result = [seed_text]

  # Process seed text char by char because generate_one_step() expects all input to be same length
  for seed_char in seed_text:
    seed_char_tensor = tf.constant([seed_char])
    _, states = one_step_model.generate_one_step(seed_char_tensor, states = states)

  next_char_tensor = tf.constant([seed_text[-1]]) # tensor of last char

  for _ in range(length):
    next_char_tensor, states = one_step_model.generate_one_step(next_char_tensor, states = states)
    next_char = next_char_tensor.numpy().item().decode('utf-8')
    result.append(next_char)

  return tf.strings.join(result)

# Test
generated_text = generate_text("to be or not to be")
print(generated_text)