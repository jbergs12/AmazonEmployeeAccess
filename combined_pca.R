library(tidymodels)
library(tidyverse)
library(vroom)
library(glmnet)
library(embed)
source("amz_recipe.R")

amz_train <- vroom("train.csv")
amz_train$ACTION <- as.factor(amz_train$ACTION)
amz_test <- vroom("test.csv")

amz_rec <- amz_recipe(amz_train)



# Logistic Regression
amz_logreg <- logistic_reg() |> 
  set_engine("glm")

logreg_wf <- workflow() |> 
  add_model(amz_logreg) |> 
  add_recipe(amz_rec) |> 
  fit(data=amz_train)

logreg_preds <- logreg_wf |>
  predict(new_data = amz_test,
          type="prob")

kaggle_submission <- logreg_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION=.pred_1)

vroom_write(x=kaggle_submission, file="./logreg_pca.csv", delim = ",")



# Penalized Logistic Regression
library(lme4)

amz_plogreg <-logistic_reg(penalty = tune(),
                           mixture = tune()) |> 
  set_engine("glmnet")

plog_grid <- grid_regular(
  penalty(),
  mixture(),
  levels = 6
)

folds <- vfold_cv(amz_train, v = 5, repeats = 1)

plogreg_wf <- workflow() |> 
  add_model(amz_plogreg) |> 
  add_recipe(amz_rec)

CV_results <- plogreg_wf |> 
  tune_grid(resamples = folds,
            grid = plog_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results |> 
  select_best(metric = "roc_auc")

bestTune$penalty
bestTune$mixture

final_wf <- plogreg_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=amz_train)

plogreg_preds <- final_wf |> 
  predict(new_data = amz_test,
          type = "prob")

kaggle_submission <- plogreg_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission, file="./plogreg_pca.csv", delim = ",")



# KNN
amz_knn <- nearest_neighbor(neighbors=10) |> 
  set_mode("classification") |> 
  set_engine("kknn")

final_wf <- workflow() |> 
  add_recipe(amz_rec) |> 
  add_model(amz_knn) |>
  fit(data=amz_train)

knn_preds <- final_wf |> 
  predict(new_data = amz_test,
          type = "prob")

kaggle_submission <- knn_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission, file="./knn_pca.csv", delim = ",")



# Random Forest
library(ranger)

amz_forest <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) |> 
  set_mode("classification") |> 
  set_engine("ranger")

forest_grid <- grid_regular(
  mtry(range = c(1, 40)),
  min_n(),
  levels = 5)

forest_wf <- workflow() |> 
  add_model(amz_forest) |> 
  add_recipe(amz_rec)

folds <- vfold_cv(amz_train, v = 5, repeats = 1)

CV_results <- run_cv(forest_wf, folds, forest_grid, metric = metric_set(roc_auc),
                     cores = 10)

bestTune <- CV_results |> 
  select_best(metric = "roc_auc")

bestTune$mtry
bestTune$min_n

final_wf <- forest_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=amz_train)

forest_preds <- final_wf |> 
  predict(new_data = amz_test,
          type = "prob")

kaggle_submission <- forest_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission, file="./rforest_pca.csv", delim = ",")



# Naive Bayes
library(naivebayes)
library(discrim)

amz_nbayes <- naive_Bayes(Laplace = 0,
                          smoothness = .5) |> 
  set_mode("classification") |> 
  set_engine("naivebayes")

final_wf <- workflow() |> 
  add_recipe(amz_rec) |> 
  add_model(amz_nbayes) |>
  fit(data=amz_train)

nbayes_preds <- final_wf |> 
  predict(new_data = amz_test,
          type = "prob")

kaggle_submission <- nbayes_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission, file="./nbayes_pca.csv", delim = ",")
