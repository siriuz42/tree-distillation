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

generator <- make_uniform_sampler(c(-1, -1, -1), c(1, 1, 1))
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
y <- as.factor(y)
x <- as.numeric(as.matrix(BreastCancer[, 2:10]))
x <- matrix(as.numeric(x), ncol = 9)

df <- data.frame(x, y)
nn <- neuralnet(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9, data = df, hidden = 5, linear.output = FALSE)
yy <- nn$net.result[[1]]
yhat <- yy[, 1] < yy[, 2]
print(table(y, yhat))
plot(nn)

teacher <- function(data) {
  predict(nn, data)
}

make_bc_sampler <- function(x, sigma=0.1) {
  bc_sampler <- function(n, lower=NULL, upper=NULL){
    if (is.null(lower) || is.null(upper)) {
      x_sample_indices <- sample(nrow(x), n, replace=TRUE)
      result <- x[x_sample_indices, ] + matrix(
        sample(c(1, -1, 0, 0, 0, 0, 0, 0, 0, 0), n * ncol(x), replace=TRUE), 
        nrow=n,
        byrow=TRUE)
    } else {
      xt <- t(x)
      x_lower <- (xt >= lower)
      x_upper <- (xt < upper)
      x_selected <- which(apply((x_lower & x_upper), 2, all))
      if (is.null(x_selected) || (length(x_selected) == 0)) {
        warning('No example satisfying the bound condition.')
        result <- matrix(runif(n * ncol(x)) * (upper - lower) + lower,
                         nrow=n, byrow=TRUE)
      } else {
        x_sample_indices <- sample(x_selected, n, replace=TRUE)
        raw <- x[x_sample_indices, ] + matrix(
          sample(c(1, -1, 0, 0, 0, 0, 0, 0, 0, 0), n * ncol(x), replace=TRUE), 
          nrow=n,
          byrow=TRUE)
        result <- t(pmax(pmin(t(raw), upper), lower))
      }
    }
    result
  }
  bc_sampler
}

### needs generator
generator <- make_bc_sampler(x)
mimic_tree <- distillation_tree(
  teacher = teacher,
  generator = generator,
  max_sample_size = 50000,
  max_stepsize = 30000,
  min_stepsize = 10000,
  stop_tree_depth = 4
)

mimic_tree_sameforest = list()
mimic_tree_sameforest_base = list()
for (i in 1:100) {
  cat(i)
  mimic_tree_sameforest[[i]] <- distillation_tree(
    teacher=teacher,
    generator=generator,
    max_sample_size=50000,
    max_stepsize=30000,
    min_stepsize=10000,
    init_sample_size=10000,
    stop_tree_depth=4)
}
for (i in 1:100) {
  cat(i)
  baseline_x <- generator(10000)
  mimic_tree_sameforest_base[[i]] <- distillation_tree(
    teacher=teacher,
    generator=generator,
    max_sample_size=50000,
    max_stepsize=30000,
    min_stepsize=10000,
    init_sample_size=10000,
    stop_tree_depth=4,
    baseline_x=baseline_x)
}

save(mimic_tree_sameforest, file="bc_nn_tree_big.RData")
load("bc_nn_tree_big.RData")
plot_trees(mimic_tree_sameforest, name="AppTree")
save(mimic_tree_sameforest_base, file="bc_nn_tree_base_big.RData")
load("bc_nn_tree_base_big.RData")
plot_trees(mimic_tree_sameforest_base, name="BASE")

par(mfrow=c(1,4))

require(gridExtra)

load("bc_nn_tree.RData")
p1 <- summarize_trees(mimic_tree_sameforest, 
                      name="Breast Cancer Nps=5,000",
                      x_tick="AppTree")
load("bc_nn_tree_base.RData")
p2 <- summarize_trees(mimic_tree_sameforest_base, 
                      name="Breast Cancer Nps=5,000",
                      x_tick="BASE")
load("bc_nn_tree_big.RData")
p3 <- summarize_trees(mimic_tree_sameforest, 
                      name="Breast Cancer Nps=50,000",
                      x_tick="AppTree")
load("bc_nn_tree_base_big.RData")
p4 <- summarize_trees(mimic_tree_sameforest_base, 
                      name="Breast Cancer Nps=50,000",
                      x_tick="BASE")
sheet <- rbind(p2, p1, p4, p3)
png("bc_nn.png")
plot_trees(sheet)
dev.off()
