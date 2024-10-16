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

vroom_write(x=kaggle_submission, file="./logreg2.csv", delim = ",")
