library(tidymodels)
library(tidyverse)
library(vroom)
library(glmnet)
source("amz_recipe.R")

amz_train <- vroom("train.csv")
amz_test <- vroom("test.csv")

amz_rec <- amz_recipe(amz_train)

amz_plogreg <-logistic_reg(penalty = tune(),
                           mixture = tune()) |> 
  set_engine("glm")

plog_tune <- grid_regular(
  penalty(),
  mixture()
)

plogreg_wf <- workflow() |> 
  add_model(amz_plogreg) |> 
  add_recipe(amz_rec)