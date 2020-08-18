library(rpart)
library(rpart.plot)
library(randomForest)

source("StoppingRule.R")


#### PART 1 : Two Level Tree ####
n <- 500
X <- matrix(runif(n * 2), ncol = 2)
y <- rep(0, n)

y[X[, 1] < 0.5] <- -1 + rnorm(sum(X[, 1] < 0.5))
y[X[, 1] >= 0.5] <- 1 + rnorm(sum(X[, 1] >= 0.5))

rf <- randomForest(X, y, ntree = 10000, sampsize = 200, replace = FALSE, keep.inbag = TRUE)

n_stu <- 10000
X_stu <- matrix(runif(n_stu * 2), ncol = 2)
y_stu <- predict(rf, X_stu)

tree <- rpart(y_stu ~ X_stu, cp = 0.005)
rpart.plot(tree)

### root node ###
n_test <- 10
X_test <- matrix(runif(n_test * 2), ncol = 2)

ptestu(X_test, rf)
# det(calc_var(X_test, rf, correction = TRUE)$var)
# ptestu(X_test, rf, correction = FALSE)
# det(calc_var(X_test, rf, correction = FALSE)$var)

### left child node ###
n_test <- 10
X_test <- matrix(runif(5 * n_test * 2), ncol = 2)
X_test <- X_test[X_test[, 1] < 0.5, ][1:n_test, ]

ptestu(X_test, rf)
# ptestu(X_test, rf, correction = FALSE)

### right child node ###
n_test <- 10
X_test <- matrix(runif(5 * n_test * 2), ncol = 2)
X_test <- X_test[X_test[, 1] >= 0.5, ][1:n_test, ]

ptestu(X_test, rf)
# ptestu(X_test, rf, correction = FALSE)


#### PART 2 : MARS Data ####
n <- 500
X <- matrix(runif(n * 5), ncol = 5)
eps <- rnorm(n, sd = 0.05)
y <- 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.05)^2 + 10 * X[, 4] + 5 * X[, 5] + eps

n_test <- 5
set.seed(7)
X_test <- matrix(runif(n_test * 5), ncol = 5)

rf <- randomForest(X, y, ntree = 50000, sampsize = 100, replace = FALSE, keep.inbag = TRUE)
ans <- calc_var(X_test, rf, correction = TRUE)
ans$var

ptestu(X_test, rf)
# ptestu(X_test, rf, correction = FALSE)
