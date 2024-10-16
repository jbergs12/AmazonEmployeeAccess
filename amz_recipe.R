amz_recipe <- function(traindata){
  recipe(ACTION~., data=traindata) |> 
    step_mutate_at(all_numeric_predictors(), fn = factor) |>
    step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
    #step_zv(all_predictors()) |> 
    step_normalize(all_numeric_predictors())
}

run_cv <- function(wf, folds, grid, metric=metric_set(rmse), cores=8){
  library(doParallel)
  
  cl <- makePSOCKcluster(cores)
  registerDoParallel(cl)
  
  results <- wf |>
    tune_grid(resamples=folds,
              grid=grid,
              metrics=metric)
  
  stopCluster(cl)
  return(results)
}