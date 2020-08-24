source("distillation_tree.R")
library(randomForest)
library(neuralnet)
library(mlbench)

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


#### BreastCancer Data ####

data("BreastCancer")

#Clean off rows with missing data
BreastCancer = BreastCancer[which(complete.cases(BreastCancer) == TRUE), ]

y <- as.matrix(BreastCancer[, 11])
y[which(y == "benign")] <- 0
y[which(y == "malignant")] <- 1
y <- as.numeric(y)
x <- as.numeric(as.matrix(BreastCancer[, 2:10]))
x <- matrix(as.numeric(x), ncol = 9)

df <- data.frame(cbind(x, y))
nn <- neuralnet(y ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = df, hidden = 5, linear.output = FALSE)
yy <- nn$net.result[[1]]
yhat <- matrix(0, length(y), 1)
yhat[which(yy > mean(yy))] <- 1
yhat[which(yy <= mean(yy))] <- 0

print(table(y, yhat))
plot(nn)

teacher <- function(data) {
  predict(nn, data)
}

### needs generator

generator <- NULL
mimic_tree <- distillation_tree(
  teacher = teacher,
  generator = generator,
  max_sample_size = 50000,
  max_stepsize = 30000,
  min_stepsize = 10000,
  stop_tree_depth = 4
)





