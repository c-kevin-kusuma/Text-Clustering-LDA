---
title: "Use Case Discovery"
author: "Kevin Kusuma"
date: "1/26/2021"
output:
  html_document: default
  pdf_document: default
---
Notes:
1. There are two main parts of this project: Topic and Predictive modelings
2. Topic modeling is used to cluster and label each topic
3. AutoML on DOMO is used to create the predictive model
4. I didn't know that Quanteda package wasn't available on DOMO so I had to use 
   Tidytext to create the live prediction input

```{r setup, warning=FALSE, message=FALSE}
options("scipen"=8, "digits"=3)
options(tinytex.verbose = TRUE)
domo <- rdomo::Domo()
library(tidyverse)
library(tidytext)
library(quanteda)
library(topicmodels)
library(janitor)
library(doSNOW)
cl <- makeCluster(8, type = "SOCK")
library(caret)
library(caTools)
library(SnowballC)
library(tm)
```

# Page & Card Data
```{r, message=FALSE, warning=FALSE}
# Run this commented-out codes to get the most updated dataset
# page_card_data <- domo$ds_get('5ff72aca-c214-4b72-8974-b19e4a7ec762') %>%
#   select(`instance_name`, environment, page_id, page_card_names) %>%
#   clean_names()

# Run this code if you want to get the original raw data
page_card_data <- domo$ds_get('0b80edf4-872a-4390-ad65-1341363679e6')
```

# Tokenizations using Quanteda Package
```{r}
tkn_data <- 
  tokens(page_card_data$page_card_names, what = 'word',
         remove_numbers = TRUE, remove_punct = TRUE,
         remove_symbols = TRUE, split_hyphens = TRUE)
```


# Clean the tokenized data
```{r}
# A list of unnecessary words
removed_words <- 
  append(stopwords(), 
         c('na', 'month', 'monthly', 'data', 'week', 'weekly', 'total', 'day', 
           'daili', 'year', 'yearly', 'date', 'ytd', 'mtd', 'current', 'averag', 
           'test', 'dashboard', 'hour', 'yoy','mom','dataset','avg','sum','card', 
           'count', 'filter', 'previous', 'histor', 'histori', 'prod', 'report', 
           'master', 'chang', 'overall', 'output', 'etl','v2','v3', 'list','ly',
           'fy20', 'cy','today','yesterday', 'categori', 'fileupload', 'detail',
           'sql', 'full', 'googlesheet', 'q1', 'excel', 'dataflow', 'raw', 'q3',
           'bi_kbi', 'fileuploadnew', 'quarter','mql', 'qtd', 'q4', 'q1', 'q2', 
           'dtv_data_masterdeal_v2', 'fy21', 'fy20', 'fy19', 'kpi', 'kpis'))
                
lda_data <- tkn_data %>% 
  tokens_tolower() %>%  # Apply Lower()
  tokens_wordstem() %>% # Transform the words to their root
  tokens_remove(removed_words) %>% # Remove unnecessary words
  dfm() %>% # Create Document Term Matrix
  dfm_trim(min_docfreq = 20, 
           termfreq_type = 'count') # Only include words in >= 20 pages

# This key table is to store the primary key fields
key <- data.frame(doc_id = rownames(lda_data), 
                  select(page_card_data, c(instance_name, environment, page_id)))

# Remove documents without meaningful words
row_sum <- apply(lda_data, 1, FUN = sum)
lda_data <- lda_data[row_sum!=0, ]
```

# Latent Dirichlet Allocation (Topic Model)
```{r}
# Running LDA with 20 Topics
set.seed(212)
LDA_1 <- LDA(lda_data, 20, 
             method = 'Gibbs', 
             control = list(iter=500, verbose=100))

LDA_1_result <- posterior(LDA_1)

# Terms
LDA_1_best_terms <- terms(LDA_1, 50) %>% data.frame() %>% 
  rownames_to_column('rank') %>% mutate(type = 'lda_1_model') %>% 
  pivot_longer(cols = 2:21, names_to = 'topic', values_to = 'term') %>% 
  mutate(topic = as.integer(sub('Topic.', '', topic)))


# Topics
LDA_1_topics <- LDA_1_result$topics %>% data.frame() %>% rownames_to_column('doc_id')
topic_longer <- pivot_longer(LDA_1_topics, cols = c(2:ncol(LDA_1_topics)), 
                             names_to = 'topic', values_to = 'prob')
topic_max <- data.frame(doc_id = LDA_1_topics$doc_id, 
             max = apply(LDA_1_topics[ ,2:ncol(LDA_1_topics)], 1, max))
LDA_1_topics <- left_join(topic_longer, topic_max) %>%
  filter(prob == max) %>% select(-max)

# Remove pages that have multiple topic assignments
weak_page_1 <- LDA_1_topics %>% 
  group_by(doc_id) %>% 
  summarise(topic = n_distinct(topic)) %>% 
  filter(topic >1)

LDA_1_topics <- anti_join(LDA_1_topics, weak_page_1, by = c("doc_id"="doc_id"))
LDA_1_topics <- inner_join(key, LDA_1_topics, by = c("doc_id"="doc_id"))

best_pages <- LDA_1_topics %>% 
  group_by(topic) %>% 
  mutate(rank = dense_rank(desc(prob))) %>% 
  arrange(topic, rank) %>% 
  filter(rank<=700)
```

# Modeling Data Push to DOMO for Visualization and Analysis
```{r}
# # Term Data Push - "Use Case Discovery | Terms"
# domo$ds_update('b9a15684-e76e-4c2e-9733-36fbcbf81d86', LDA_1_best_terms)
# 
# # Topic Data Push - "Use Case Discovery | Topics"
# domo$ds_update('d00826c3-a394-480f-83fc-c152a60cc6ed', LDA_1_topics)
```

# Predictive Portion
```{r}
# A list of unnecessary words
removed_words <- rbind(
  data.frame(word = c('na', 'total', 'prod', 'ー', '＿'), lexicon = 'SMART'), 
  stop_words) %>% 
  anti_join(data.frame(word = unique(LDA_1_best_terms$term)))

# Tokenizations using tidytext package
training <- 
  page_card_data[,-1] %>% 
  unnest_tokens(
    output = word, 
    input = page_card_names,
    token = 'words',
    to_lower = TRUE,
    format = 'text') %>%
  mutate(word = removePunctuation(word),
         word = wordStem(word)) %>% 
  anti_join(removed_words) %>% 
  filter(word != '') %>% 
  mutate(value = 1) %>% 
  group_by(instance_name, environment, page_id, word) %>% 
  summarise(value = sum(value))

# A list of best terms from the Quanteda package that can't be found in tidytext
missing_best_terms <- 
  anti_join(LDA_1_best_terms, training[,4], by = c("term"="word")) %>% 
  arrange(as.integer(rank))

training_smaller <- 
  inner_join(training, data.frame(word = unique(LDA_1_best_terms$term)), 
             by = c("word"="word")) %>% inner_join(LDA_1_topics[,2:5]) %>% 
  rename(environments = environment, lda_topic = topic)

# Training and Testing Assignment
set.seed(212)
partition_data <- 
  training_smaller %>% 
  group_by(instance_name, environments, page_id, lda_topic) %>% 
  summarise(value = sum(value))

partition <- createDataPartition(partition_data$lda_topic, p = 0.7, list = FALSE)
train_rows <- cbind(partition_data[partition, -5], data_type = 'training')

# Assign the training and testing tags
training_bigger <- 
  training_smaller %>% left_join(train_rows) %>% 
  mutate(data_type = replace_na(data_type, 'testing'))

# Widen the data
training_input <- 
  pivot_wider(training_bigger, 
              id_cols = c(1,2,3,6,7), 
              names_from = word, 
              values_from = value, 
              values_fill = 0)

# Push the training data to DOMO - "Use Case Discovery | Training Data"
# domo$ds_update('079cb3d6-44ad-422b-ae03-ed0f8925485d', training_input)
```

# AutoML result
```{r}
# AutoML Result Data - "Use Case Discovery | Predictive Model"
auto_ml <- domo$ds_query('88722495-850d-47f6-a0ef-1c8b48e9790e',
                         'select * from table')

auto_ml %>% filter(data_type != 'new') %>%  group_by(probs = round(probs, 1)) %>% 
  summarise(accuracy = mean(ifelse(prediction == lda_topic, 1, 0))) %>% 
  ggplot(aes(x = probs, y = accuracy)) + geom_col() + 
  geom_text(aes(label = round(accuracy, 2)))

quantile(auto_ml$probs)
hist(auto_ml$probs)
```
