https://blogs.rstudio.com/ai/posts/2017-12-22-word-embeddings-with-keras/
https://rpubs.com/nabiilahardini/word2vec
https://cbail.github.io/textasdata/word2vec/rmarkdown/word2vec.html
https://www.theanalyticslab.nl/nlpblogs_2_training_word_embedding_models_and_visualize_results/
https://medium.com/cmotions/natural-language-processing-for-predictive-purposes-with-r-cb65f009c12b

download.file("https://bhciaaablob.blob.core.windows.net/cmotionsnlpblogs/reviews.csv",
              "reviews.csv")

download.file("https://bhciaaablob.blob.core.windows.net/cmotionsnlpblogs/labels.csv",
              "labels.csv")

download.file("https://bhciaaablob.blob.core.windows.net/cmotionsnlpblogs/restoid.csv",
              "restoids.csv")

download.file("https://bhciaaablob.blob.core.windows.net/cmotionsnlpblogs/trainids.csv",
              "trainids.csv")

# Read data files from public blob storage
