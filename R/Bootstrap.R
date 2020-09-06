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

x <- x.generator(1000)
sampler_1 <- make_bootstrap_sampler(x)
sampler_2 <- make_gaussian_sampler(x, sigma=0.2)
y <- as.factor(y.generator(x))

forest <- randomForest(x, y, sampsize=500, 
                       replace = FALSE, 
                       keep.inbag = TRUE, 
                       ntree = 100)

teacher <- function(data) {
  predict(forest, newdata=data, type="prob")
}

mimic_trees_1 = list()
mimic_trees_2 = list()
for (i in 1:100) {
  cat(i, '\n')
  mimic_trees_1[[i]] <- distillation_tree(teacher=teacher,
                                          generator=sampler_1,
                                          max_sample_size=10000,
                                          max_stepsize=3000,
                                          min_stepsize=1000,
                                          init_sample_size=1000,
                                          stop_tree_depth=4)
  mimic_trees_2[[i]] <- distillation_tree(teacher=teacher,
                                          generator=sampler_2,
                                          max_sample_size=10000,
                                          max_stepsize=3000,
                                          min_stepsize=1000,
                                          init_sample_size=1000,
                                          stop_tree_depth=4)
}
save(mimic_trees_1, file="bs_1.RData")
load("bs_1.RData")
save(mimic_trees_2, file="bs_2.RData")
load("bs_2.RData")

#### PART 2 : Synthetic Data 2 ####

set.seed(42)
x.generator <- function(n, d = 5) {
  return(matrix(runif(n*d, -1, 1), nrow=n))
}

y.generator <- function(x) {
  return(as.numeric(
    runif(nrow(x)) < sigmoid(-3 * (x[,3] * x[, 4]) * (x[, 1]<0) + 3 * x[, 2] * (x[, 1] > 0))
  ))
}


x <- x.generator(1000)
sampler_3 <- make_bootstrap_sampler(x)
sampler_4 <- make_gaussian_sampler(x, sigma=0.2)
y <- as.factor(y.generator(x))

forest <- randomForest(x, y, sampsize = 500, 
                       replace = FALSE, 
                       keep.inbag = TRUE, 
                       ntree = 100)

teacher <- function(data) {
  predict(forest, newdata=data, type="prob")
}

mimic_trees_3 = list()
mimic_trees_4 = list()
for (i in 1:100) {
  cat(i, '\n')
  mimic_trees_3[[i]] <- distillation_tree(teacher=teacher,
                                          generator=sampler_3,
                                          max_sample_size=50000,
                                          max_stepsize=30000,
                                          min_stepsize=10000,
                                          init_sample_size=10000,
                                          stop_tree_depth=4)
  mimic_trees_4[[i]] <- distillation_tree(teacher=teacher,
                                          generator=sampler_4,
                                          max_sample_size=50000,
                                          max_stepsize=30000,
                                          min_stepsize=10000,
                                          init_sample_size=10000,
                                          stop_tree_depth=4)
}
save(mimic_trees_3, file="bs_3.RData")
load("bs_3.RData")
save(mimic_trees_4, file="bs_4.RData")
load("bs_4.RData")


p1 <- summarize_trees(mimic_trees_1, 
                      name="Synthetic Data 1",
                      x_tick="Bootstrap")
p2 <- summarize_trees(mimic_trees_2, 
                      name="Synthetic Data 1",
                      x_tick="Gaussian Density")
p3 <- summarize_trees(mimic_trees_3, 
                      name="Synthetic Data 2",
                      x_tick="Bootstrap")
p4 <- summarize_trees(mimic_trees_4, 
                      name="Synthetic Data 2",
                      x_tick="Gaussian Density")
sheet <- rbind(p1, p2, p3, p4)
png("bootstrap.png")
plot_trees(sheet)
dev.off()