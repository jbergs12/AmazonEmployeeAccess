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

vroom_write(x=kaggle_submission, file="./plogreg.csv", delim = ",")
