library(tidymodels)
library(vroom)
library(embed)

source("amz_recipe.R")

amz_train <- vroom("train.csv")
amz_train$ACTION <- as.factor(amz_train$ACTION)
amz_test <- vroom("test.csv")

amz_rec <- amz_recipe(amz_train)

amz_knn <- nearest_neighbor(neighbors=tune()) |> 
  set_mode("classification") |> 
  set_engine("kknn")

knn_grid <- grid_regular(
  neighbors(),
  levels = 10
)

knn_wf <- workflow() |> 
  add_model(amz_knn) |> 
  add_recipe(amz_rec)

folds <- vfold_cv(amz_train, v = 10, repeats = 1)

CV_results <- run_cv(knn_wf, folds, knn_grid, metric = metric_set(roc_auc),
                     cores = 7)

bestTune <- CV_results |> 
  select_best(metric = "roc_auc")

bestTune$neighbors # 10

final_wf <- knn_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=amz_train)

knn_preds <- final_wf |> 
  predict(new_data = amz_test,
          type = "prob")

kaggle_submission <- knn_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission, file="./knn.csv", delim = ",")
