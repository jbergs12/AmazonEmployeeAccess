library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
source("amz_recipe.R")

library(ggmosaic)

amz_train <- vroom("train.csv")
amz_test <- vroom("test.csv")

# EDA
amz_train |> 
  ggplot() +
  geom_mosaic(aes(x=product(as.factor(ROLE_DEPTNAME), fill = ACTION)))


# Recipe

baked_rec <- bake(prep(amz_recipe(amz_train)), new_data = amz_train)

ncol(baked_rec)
