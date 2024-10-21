library(tidymodels)
library(vroom)
library(embed)
library(ranger)

source("amz_recipe.R")

amz_train <- vroom("train.csv")
amz_train$ACTION <- as.factor(amz_train$ACTION)
amz_test <- vroom("test.csv")

amz_rec <- amz_recipe(amz_train)

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

vroom_write(x=kaggle_submission, file="./rforest.csv", delim = ",")
