## Bootstrap Simulation
## Refer to sim_190817

source("distillation_tree.R")
library(randomForest)

sigmoid <- function(x) return(1/(1+exp(-x)))

#### PART 1 : Synthetic Data 1 ####
set.seed(42)

x.generator <- function(n, d=3) {
  matrix(runif(n*d, -1, 1), nrow=n)
}

y.generator <- function(x) {
  as.numeric(runif(nrow(x)) < sigmoid(
    -3 * (x[, 1] < 0) + 3 * x[, 2] * (x[, 1] > 0)))
}

x <- x.generator(10000)
sampler <- make_bootstrap_sampler(x)
y <- as.factor(y.generator(x))

forest <- randomForest(x, y, sampsize = 1000, 
                       replace = FALSE, 
                       keep.inbag = TRUE, 
                       ntree = 1000)

teacher <- function(data) {
  predict(forest, newdata=data, type="prob")
}

mimic_tree <- distillation_tree(teacher=teacher,
                                generator=sampler,
                                max_sample_size=50000,
                                max_stepsize=30000,
                                min_stepsize=10000,
                                stop_tree_depth=4)
save(mimic_tree, file="test_single.RData")

mimic_tree_sameforest = list()
for (i in 1:100) {
  mimic_tree_sameforest[[i]] <- distillation_tree(teacher=teacher,
                                                  generator=sampler,
                                                  max_sample_size=50000,
                                                  max_stepsize=30000,
                                                  min_stepsize=10000,
                                                  stop_tree_depth=4)
}
save(mimic_tree_sameforest, file="test_multiple.RData")
