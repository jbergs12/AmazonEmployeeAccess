library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
source("amz_recipe.R")

library(ggmosaic)

amz_train <- vroom("train.csv")
amz_test <- vroom("test.csv")

# EDA

table(amz_train$ROLE_DEPTNAME, amz_train$ACTION)

table(amz_train$ROLE_TITLE, amz_train$ACTION)

amz_train |> 
  ggplot() +
  geom_mosaic(aes(x=product(as.factor(ROLE_DEPTNAME),
                            fill = as.factor(ACTION))))

amz_train |> 
  ggplot(aes(x=as.factor(RESOURCE),
             y=as.factor(ACTION))) +
  geom_boxplot()


# Recipe

baked_rec <- bake(prep(amz_recipe(amz_train)), new_data = amz_train)

ncol(baked_rec)
