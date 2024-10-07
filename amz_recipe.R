amz_recipe <- function(traindata){
  recipe(ACTION~., data=traindata) |> 
    step_mutate_at(all_numeric_predictors(), fn = factor) |> 
    step_other(all_predictors(), threshold = .001) |> 
    step_dummy(all_predictors())
    # step_lencode_mixed(all_predictors(), outcome = vars(ACTION))
}