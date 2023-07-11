# sumber
# https://rpubs.com/nabiilahardini/word2vec
setwd("~/LLM-ikanx/R/artikel 3")
rm(list=ls())

library(tidyverse)
library(parallel)

n_cot = 5

csvs   = list.files(pattern = "csv")
import = mclapply(csvs,read_csv,mc.cores = n_cot)

data = do.call(rbind,import) %>% distinct()
stopwords = readLines("https://raw.githubusercontent.com/ikanx101/ID-Stopwords/master/id%20stopwords%20modif.txt")

data_clean <- data %>% 
  mutate(text = text %>% 
           
           # turn text into lowercase
           str_to_lower() %>% 
           # remove stopwords
           tm::removeWords(words = stopwords) %>%
           # reduce repeated whitespace from the text
           str_squish()) %>% 
  mutate(title = title %>% 
           
           # turn title into lowercase
           str_to_lower() %>% 
           # remove stopwords
           tm::removeWords(words = stopwords) %>%
           # reduce repeated whitespace from the title
           str_squish()) %>% 
  mutate(text = gsub("[[:punct:]]"," ",text))


# Load packages
library(stringdist)
library(stringr)
library(dplyr)
library(tidyr)
library(tidytext)
library(sjmisc)
library(vroom)
library(keras)
library(reticulate)
library(purrr)


# Create text vector ("text_data")
text_data <- data_clean$text

# prepare for NLP with keras 
tokenizer <- text_tokenizer(num_words = 2472) # create token for 20,000 most common words in data
tokenizer %>% fit_text_tokenizer(text_data)

# define a generator function for model training 
skipgrams_generator <- function(text, tokenizer, window_size, negative_samples) {
  gen <- texts_to_sequences_generator(tokenizer, sample(text))
  function() {
    skip <- generator_next(gen) %>%
      skipgrams(
        vocabulary_size = tokenizer$num_words, 
        window_size = window_size, 
        negative_samples = 1
      )
    x <- transpose(skip$couples) %>% map(. %>% unlist %>% as.matrix(ncol = 1))
    y <- skip$labels %>% as.matrix(ncol = 1)
    list(x, y)
  }
}

# define variables 
embedding_size <- 256  # Dimension of the embedding vector.
skip_window <- 5       # How many words to consider left and right.
num_sampled <- 1       # Number of negative examples to sample for each word.
input_target <- layer_input(shape = 1)
input_context <- layer_input(shape = 1)

# create the embedding matrix
embedding <- layer_embedding(
  input_dim = tokenizer$num_words + 1, 
  output_dim = embedding_size, 
  input_length = 1, 
  name = "embedding"
)

target_vector <- input_target %>% 
  embedding() %>% 
  layer_flatten()

context_vector <- input_context %>%
  embedding() %>%
  layer_flatten()

# use dot product to calculate text similarity 
dot_product <- layer_dot(list(target_vector, context_vector), axes = 1)
output <- layer_dense(dot_product, units = 1, activation = "sigmoid")

# create and run the model 
model <- keras_model(list(input_target, input_context), output)
model %>% compile(loss = "binary_crossentropy", optimizer = "adam")

model %>%
  keras::fit(
    skipgrams_generator(text_data, tokenizer, skip_window, num_sampled), 
    steps_per_epoch = 3, epochs = 7, verbose = 1
  )

#obtaining word vector
embedding_matrix <- get_weights(model)[[1]]

words <- data.frame(
  word = names(tokenizer$word_index), 
  id = as.integer(unlist(tokenizer$word_index))
)

words <- words %>%
  dplyr::filter(id <= tokenizer$num_words) %>%
  dplyr::arrange(id)


words %>% head(5)
row.names(embedding_matrix) <- c("UNK", words$word)

dim(embedding_matrix)


#install.packages("text2vec")
library(text2vec)
find_similar_words <- function(word, embedding_matrix, n = 5) {
  similarities <- embedding_matrix[word, , drop = FALSE] %>%
    sim2(embedding_matrix, y = ., method = "cosine")
  similarities[,1] %>% sort(decreasing = TRUE) %>% head(n)
}
find_similar_words("desain", embedding_matrix)
