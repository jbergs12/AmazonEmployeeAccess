library(tidymodels)
library(tidyverse)
library(vroom)
library(glmnet)
library(embed)
source("amz_recipe.R")

amz_train <- vroom("train.csv")
amz_train$ACTION <- as.factor(amz_train$ACTION)
amz_test <- vroom("test.csv")

amz_rec <- amz_smote_recipe(amz_train, neighbors=10)



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
                     cores = 5)

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

kaggle_submission_rf <- forest_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission_rf, file="./rforest_smote.csv", delim = ",")



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

kaggle_submission_lr <- logreg_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION=.pred_1)

vroom_write(x=kaggle_submission_lr, file="./logreg_smote.csv", delim = ",")



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

CV_results_plr <- run_cv(plogreg_wf, folds, plog_grid, metric = metric_set(roc_auc),
                     parallel=FALSE)

bestTune_plr <- CV_results_plr |> 
  select_best(metric = "roc_auc")

bestTune_plr$penalty
bestTune_plr$mixture

final_wf <- plogreg_wf |> 
  finalize_workflow(bestTune_plr) |> 
  fit(data=amz_train)

plogreg_preds <- final_wf |> 
  predict(new_data = amz_test,
          type = "prob")

kaggle_submission_plr <- plogreg_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission_plr, file="./plogreg_smote.csv", delim = ",")



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

kaggle_submission_knn <- knn_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission_knn, file="./knn_smote.csv", delim = ",")



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

kaggle_submission_nb <- nbayes_preds |> 
  bind_cols(amz_test) |> 
  select(id, .pred_1) |> 
  rename(Id = id,
         ACTION = .pred_1)

vroom_write(x=kaggle_submission_nb, file="./nbayes_smote.csv", delim = ",")

# # SVM
# library(kernlab)
# 
# svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svmLinear <- svm_linear(cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svmRad_wf <- workflow() |> 
#   add_recipe(amz_rec) |> 
#   add_model(svmRadial)
# 
# svmRad_grid <- grid_regular(
#   rbf_sigma(),
#   cost(),
#   levels = 5)
# 
# folds <- vfold_cv(amz_train, v = 5, repeats = 1)
# 
# CV_results <- run_cv(svmRad_wf, folds, svmRad_grid, metric = metric_set(roc_auc),
#                      cores = 7, parallel = FALSE)
# 
# bestTune <- CV_results |> 
#   select_best(metric = "roc_auc")
# 
# final_wf <- svmRad_wf |> 
#   finalize_workflow(bestTune) |> 
#   fit(data=amz_train)
# 
# svmRad_preds <- final_wf |> 
#   predict(new_data = amz_test,
#           type = "prob")
# 
# kaggle_submission <- svmRad_preds |> 
#   bind_cols(amz_test) |> 
#   select(id, .pred_1) |> 
#   rename(Id = id,
#          ACTION = .pred_1)
# 
# vroom_write(x=kaggle_submission, file="./svmRad_smote.csv", delim = ",")
