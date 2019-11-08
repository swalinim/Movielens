#-----------------------------------------------------
## RMSE: compute root mean square error (RMSE)
#-----------------------------------------------------

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#-----------------------------------------------------

#-----------------------------------------------------
## Create edx set, validation set
#-----------------------------------------------------

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#-----------------------------------------------------

#-----------------------------------------------------
## Create train and test sets from edx data set
#-----------------------------------------------------

set.seed(1, sample.kind="Rounding")

# use 10% of edx data as test set

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set

test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test set back into train set

removed <- anti_join(temp, test)
train <- rbind(train, removed)

# Remove unused data to save space

rm(removed, temp, test_index) 

#-----------------------------------------------------

#-----------------------------------------------------
## Data Exploration
#-----------------------------------------------------

# Exploring train data set

train %>% as_tibble()

train %>% summarize(
  n_users=n_distinct(userId),# unique users from train data set
  n_movies=n_distinct(movieId),# unique movies from train data set
  min_rating=min(rating),  # the lowest rating 
  max_rating=max(rating) # the highest rating
)

# Display movies and users as a matrix

movie_matrix <- train %>% 
  count(movieId) %>% 
  top_n(5, n) %>% 
  .$movieId

final_matrix <- train %>% 
  filter(movieId%in%movie_matrix) %>% 
  filter(userId %in% c(1:10)) %>% 
  select(userId, title, rating) %>% 
  mutate(title = str_remove(title, ", The"),
         title = str_remove(title, ":.*")) %>%
  spread(title, rating)

final_matrix %>% knitr::kable()


# This matrix displays a random sample of 100 movies and 100 users with yellow 
# indicating a user/movie combination for which we have a rating.

users <- sample(unique(train$userId), 100)
train %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

# This plot displays rating count by movie

train %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "blue") + 
  scale_x_log10() + 
  ggtitle("Movies")

# This plot displays rating count by users

train %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "red") + 
  scale_x_log10() + 
  ggtitle("Users")

#-----------------------------------------------------

#-----------------------------------------------------
## Building models
#-----------------------------------------------------

#****************************
# Calculate just the average
#****************************

# Calculate the average movie rating mu

mu <- mean(train$rating) 
mu

# Compute RMSE for just the average

naive_rmse <- RMSE(test$rating, mu) 
naive_rmse

# Create a results table to record RMSE for all models

rmse_results <- tibble(Method = "Just the average", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

#****************************
# Modeling movie effect
#****************************

# Fitting using least squares estimates will take a long time

#fit <- lm(rating ~ as.factor(movieId), data = train)

# We will use the fact that b_i = Y(u,i) - mu

mu <- mean(train$rating) 
movie_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))


# Calulate RMSE for model with movie effect

predicted_ratings <- mu + test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_effect <- RMSE(test$rating, predicted_ratings)

# Add this result to the results table

rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Model with Movie Effect",  
                                 RMSE = movie_effect ))
rmse_results %>% knitr::kable()

#********************************
# Modeling movie and User effect
#********************************

# The plot below shows us there is variability across users

train %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "blue")

# Fitting using least squares estimates will take a long time

##fit <- lm(rating ~ as.factor(movieId) + as.factor(userId))

# We will use the fact that b_u = Y(u,i) - mu - b_i

user_avgs <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Calulate RMSE for model with movie and user effect

predicted_ratings <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

user_effect <- RMSE(predicted_ratings, test$rating)

# Add this result to the results table

rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Model with Movie + User Effects",  
                                 RMSE = user_effect ))
rmse_results %>% knitr::kable()

#****************************
# Regularisation
#****************************

# Create a database that connects movieId to movie title

movie_titles <- train %>% 
  select(movieId, title) %>%
  distinct()

# Top 10 best movies based on b_i

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

# Top 10 worse movies based on b_i

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

# To find how often the best obscure movies are rated

train %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# To find how often the worse obscure movies are rated

train %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# use cross-validation to pick the penalty term lambda:

lambda <- seq(0, 10, 0.25)

rmses <- sapply(lambda, function(l){
  mu <- mean(train$rating)
  
  b_i <- train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    train %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(train$rating, predicted_ratings))
})

qplot(lambda, rmses)  

# pick lambda with minimun rmse
     
lambda <- lambda[which.min(rmses)]

# print lambda

lambda

# compute movie effect with regularization on train set

b_i <- train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# compute user effect with regularization on train set

b_u <- train %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# compute predicted values on test set 

predicted_ratings <- 
  test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# create a results table with this and previous approaches

model_regularization <- RMSE(test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Model with Regularized Movie and User Effect",  
                                 RMSE = model_regularization))
rmse_results %>% knitr::kable()

#**********************************************
# Matrix factorization using recosystem Package
#**********************************************

# Install/Load recosystem

if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

# The data file for training and test set needs to be arranged in sparse matrix  
# triplet form, i.e., each line in the file contains three numbers
# user_index item_index rating

train_matrix <- data_memory(user_index = train$userId, item_index = train$movieId, 
                          rating = train$rating, index1 = T)

test_matrix <- data_memory(user_index = test$userId, item_index = test$movieId, index1 = T)

# Create a model object (a Reference Class object in R) by calling Reco()

rec <- Reco()

# Call the $tune() method to select best tuning parameters

opts = rec$tune(train_matrix, opts = list(dim = c(10, 20, 30), lrate = c(0.05, 0.1, 0.2),
                                     costp_l1 = 0, costq_l1 = 0,
                                     nthread = 2))

# Display best tuning parameters 

print(opts$min)

#Train the model by calling the $train() method. A number of parameters can be set 
#inside the function, possibly coming from the result of $tune()

set.seed(1, sample.kind="Rounding")
rec$train(train_matrix, opts = c(dim = 30, costp_l1 = 0, costp_l2 = 0.01, 
                              costq_l1 = 0,costq_l2 = 0.1, lrate = 0.1,
                              verbose = FALSE))

 # Use the $predict() method to compute predicted values

predicted_ratings <- rec$predict(test_matrix, out_memory()) 

# Create a results table with matrix factorization

factorization <- RMSE(test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Model with Matrix Factorization",  
                                 RMSE = factorization))
rmse_results %>% knitr::kable()

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
## Evaluating on validation data set
#------------------------------------------------------------------------------

# The data file for edx and validation set needs to be arranged in sparse matrix  
# triplet form, i.e., each line in the file contains three numbers
# user_index item_index rating

edx_matrix <- data_memory(user_index = edx$userId, item_index = edx$movieId, 
                            rating = edx$rating, index1 = T)

validation_matrix <- data_memory(user_index = validation$userId, item_index = validation$movieId, index1 = T)

# Create a model object (a Reference Class object in R) by calling Reco()

rec_final <- Reco()

# Call the $tune() method to select best tuning parameters along a set of candidate values

opts = rec_final$tune(edx_matrix, opts = list(dim = c(10, 20, 30), lrate = c(0.05, 0.1, 0.2),
                                          costp_l1 = 0, costq_l1 = 0,
                                          nthread = 2))

# Display best tuning parameters 

print(opts$min)

#Train the model by calling the $train() method. A number of parameters can be set 
#inside the function, possibly coming from the result of $tune()

set.seed(1, sample.kind="Rounding")
rec_final$train(edx_matrix, opts = c(dim = 30, costp_l1 = 0, costp_l2 = 0.01, 
                                 costq_l1 = 0,costq_l2 = 0.1, lrate = 0.1,
                                 verbose = FALSE))

# Use the $predict() method to compute predicted values

predicted_ratings <- rec_final$predict(validation_matrix, out_memory()) 

# Create a results table with matrix factorization

factorization_final <- RMSE(validation$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Model with Matrix Factorization on validation test",  
                                 RMSE = factorization_final))
rmse_results %>% knitr::kable()

#----------------------------------------------------------------------------------






