library(tidymodels)
library(vroom)
library(embed)
library(naivebayes)
library(discrim)

source("amz_recipe.R")

amz_train <- vroom("train.csv")
amz_train$ACTION <- as.factor(amz_train$ACTION)
amz_test <- vroom("test.csv")

amz_rec <- amz_recipe(amz_train)

amz_nbayes <- naive_Bayes(Laplace = tune(),
                          smoothness = tune()) |> 
  set_mode("classification") |> 
  set_engine("naivebayes")

nbayes_wf <- workflow() |> 
  add_recipe(amz_rec) |> 
  add_model(amz_nbayes)

nbayes_grid <- grid_regular(
  Laplace(),
  smoothness(),
  levels = 5)

folds <- vfold_cv(amz_train, v = 5, repeats = 1)

CV_results <- run_cv(nbayes_wf, folds, nbayes_grid, metric = metric_set(roc_auc),
                     cores = 7)

bestTune <- CV_results |> 
  select_best(metric = "roc_auc")

bestTune$smoothness # .5
bestTune$Laplace # 0

final_wf <- nbayes_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=amz_train)

nbayes_preds <- final_wf |> 
  predict(new_data = amz_test,
          type = "prob")

kaggle_submission <- nbayes_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission, file="./nbayes.csv", delim = ",")
