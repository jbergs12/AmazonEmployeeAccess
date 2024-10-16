library(tidymodels)
library(tidyverse)
library(vroom)
library(glmnet)
library(embed)
library(lme4)

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

kaggle_submission1 <- logreg_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION=.pred_1)

# Penalized Logistic Regression

amz_plogreg <-logistic_reg(penalty = tune(),
                           mixture = tune()) |> 
  set_engine("glmnet")

plog_grid <- grid_regular(
  penalty(),
  mixture(),
  levels = 6
)

folds <- vfold_cv(amz_train, v = 10, repeats = 1)

plogreg_wf <- workflow() |> 
  add_model(amz_plogreg) |> 
  add_recipe(amz_rec)

CV_results <- run_cv(plogreg_wf, folds, plog_grid, metric = metric_set(roc_auc),
                     cores = 6)

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

kaggle_submission2 <- plogreg_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission1, file="./logreg2.csv", delim = ",")
vroom_write(x=kaggle_submission2, file="./plogreg2.csv", delim = ",")
