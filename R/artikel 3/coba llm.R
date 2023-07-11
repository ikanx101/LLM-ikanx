rm(list=ls())

# Install required packages
# install.packages("keras")

# Load necessary libraries
library(keras)
library(tidyverse)

# Prepare training data
texts <- c(
  "The cat is sitting on the mat",
  "The dog is barking loudly",
  "I love to eat pizza",
  "She sings beautifully",
  "He plays the guitar"
)

tokenizer <- text_tokenizer()
tokenizer %>% fit_text_tokenizer(texts)
sequences <- texts_to_sequences(tokenizer, texts)
vocab_size <- tokenizer$word_index %>% length()

input_sequences <- lapply(sequences, function(seq) seq[-length(seq)])
output_sequences <- lapply(sequences, function(seq) seq[-1])

input_sequences <- pad_sequences(input_sequences, maxlen = 5)
x_input_sequences <- as.matrix(input_sequences)#, num_classes = vocab_size)

output_sequences <- pad_sequences(output_sequences, maxlen = 5)
x_output_sequences <- to_categorical(output_sequences)#, num_classes = vocab_size)

# Define the model architecture
model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = ncol(input_sequences), output_dim = 5) %>%
  layer_lstm(units = 128) %>%
  layer_dense(units = vocab_size, activation = "softmax")

# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam"
)


# Train the model
model %>% keras::fit(
  x = x_input_sequences,
  y = x_output_sequences,
  batch_size = 2,
  epochs = 3,
  shuffle = TRUE
)

# Generate text
seed_text <- "The cat"
generated_text <- seed_text

for (i in 1:10) {
  encoded_text <- texts_to_sequences(tokenizer, generated_text)
  padded_text <- pad_sequences(encoded_text, maxlen = 4)
  predicted_word <- model %>% predict_classes(padded_text)
  generated_text <- paste(generated_text, tokenizer$word_index$index_word[[predicted_word]], sep = " ")
}

print(generated_text)





# =============================================================================

input_length = 30
n_sample = 5
vocab_size = 100
quest_train <- matrix(floor(runif(input_length*n_sample, 1,vocab_size)), ncol = input_length)
tag_train <- matrix(sample(c(0,1), size = input_length*n_sample, replace = T), ncol = input_length)

tag_train_reshape <- to_categorical(tag_train)

input_dim = vocab_size
embed_dim = 50


model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = input_dim,
                  output_dim = embed_dim) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 128, return_sequences = T) %>%
  layer_dropout(rate = 0.5) %>% 
  time_distributed(layer_dense(units = 2, activation = 'softmax'))

model %>%
  compile(loss = 'categorical_crossentropy', 
          optimizer = 'adam', 
          metrics = c('accuracy'))

model %>% keras::fit(quest_train,  
              tag_train_reshape, 
              batch_size = 16 ,
              epochs = 20,
              shuffle = TRUE)


