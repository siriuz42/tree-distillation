source("distillation_tree.R")
library(randomForest)
library(neuralnet())

#### Synthetic Data ####
sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}

set.seed(42)

x.generator <- function(n, d = 3) {
  matrix(runif(n * d, -1, 1), nrow = n)
}

y.generator <- function(x) {
  as.numeric(runif(nrow(x)) < sigmoid(
    -3 * (x[, 1] < 0) + 3 * x[, 2] * (x[, 1] > 0)
  ))
}

x <- x.generator(10000)
y <- as.factor(y.generator(x))

df <- data.frame(x, y)

nn <- neuralnet(y ~ ., data = df, hidden = 3, act.fct = "logistic", linear.output = FALSE)

teacher <- function(data) {
  predict(nn, data)
}

generator <- NULL
mimic_tree <- distillation_tree(
  teacher = teacher,
  generator = generator,
  max_sample_size = 50000,
  max_stepsize = 30000,
  min_stepsize = 10000,
  stop_tree_depth = 4
)
