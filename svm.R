library(tidymodels)
library(vroom)
library(embed)
library(kernlab)

source("amz_recipe.R")

amz_train <- vroom("train.csv")
amz_train$ACTION <- as.factor(amz_train$ACTION)
amz_test <- vroom("test.csv")

amz_rec <- amz_recipe(amz_train)

## SVM models
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

## Fit or Tune Model HERE (radial)

svmRad_wf <- workflow() |> 
  add_recipe(amz_rec) |> 
  add_model(svmRadial)

svmPoly_grid <- grid_regular(
  degree(),
  cost(),
  levels = 5)

svmRad_grid <- grid_regular(
  rbf_sigma(),
  cost(),
  levels = 5)

svmLin_grid <- grid_regular(
  cost(),
  levels = 5)

folds <- vfold_cv(amz_train, v = 5, repeats = 1)

CV_results <- run_cv(svmRad_wf, folds, svmRad_grid, metric = metric_set(roc_auc),
                     cores = 7, parallel = FALSE)

bestTune <- CV_results |> 
  select_best(metric = "roc_auc")

# bestTune$smoothness # .5
# bestTune$Laplace # 0

final_wf <- svmRad_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=amz_train)

svmRad_preds <- final_wf |> 
  predict(new_data = amz_test,
          type = "prob")

kaggle_submission <- svmRad_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission, file="./svmRad.csv", delim = ",")
