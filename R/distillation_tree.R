# Split rules are given by x_i < c

NUM_EXAMPLES_FOR_SPLIT_DISCOVERY <- 1000
INF <- 1e100

make_bootstrap_sampler <- function(x) {
  bootstrap_sampler <- function(n, lower=NULL, upper=NULL){
    if (is.null(lower) || is.null(upper)) {
      x_sample_indices <- sample(nrow(x), n, replace=TRUE)
    } else {
      xt <- t(x)
      x_lower <- (xt >= lower)
      x_upper <- (xt < upper)
      x_selected <- which(apply((x_lower & x_upper), 2, all))
      if (is.null(x_selected) || (length(x_selected) == 0)) {
        stop('No bootstrap example satisfying the bound condition.')
      }
      x_sample_indices <- sample(x_selected, n, replace=TRUE)
    }
    x[x_sample_indices, ]
  }
  bootstrap_sampler
}

make_gaussian_sampler <- function(x, sigma=0.1) {
  gaussian_sampler <- function(n, lower=NULL, upper=NULL){
    if (is.null(lower) || is.null(upper)) {
      x_sample_indices <- sample(nrow(x), n, replace=TRUE)
      result <- x[x_sample_indices, ] + matrix(rnorm(n * ncol(x), sd=sigma), 
                                               nrow=n)
    } else {
      xt <- t(x)
      x_lower <- (xt >= lower)
      x_upper <- (xt < upper)
      x_selected <- which(apply((x_lower & x_upper), 2, all))
      if (is.null(x_selected) || (length(x_selected) == 0)) {
        warning('No bootstrap example satisfying the bound condition.')
        result <- matrix(runif(n * ncol(x)) * (upper - lower) + lower,
                         nrow=n, byrow=TRUE)
      } else {
        x_sample_indices <- sample(x_selected, n, replace=TRUE)
        raw <- x[x_sample_indices, ] + matrix(rnorm(n * ncol(x), sd=sigma), 
                                              nrow=n)
        result <- t(pmax(pmin(t(raw), upper), lower))
      }
    }
    result
  }
  gaussian_sampler
}

get_gini <- function(x, y, splits=NULL, baseline=FALSE) {
  n_splits <- if (baseline) 1 else nrow(splits)
  n_sample <- nrow(x)
  n_class <- ncol(y)
  gini <- c()
  for (i in 1:n_splits) {
    if (baseline) {
      flag_left <- rep(TRUE, n_sample)
    } else {
      flag_left <- x[, splits[i, 1]] < splits[i, 2]
    }
    wsum <- sum(flag_left)
    if (wsum == 0) {
      tmp_left <- rbind(rep(0, n_class))
      tmp_right <- y
    } else if (wsum == n_sample) {
      tmp_left <- y
      tmp_right <- rbind(rep(0, n_class))
    } else {
      # In case flag_left is singleton, rbind keeps it a row vector.
      tmp_left <- rbind(y[flag_left,])
      tmp_right <- rbind(y[!flag_left,])
    }
    tmp_left <- apply(tmp_left, c(2), mean)
    tmp_right <- apply(tmp_right, c(2), mean)
    gini <- c(gini, 1 - wsum / n_sample * t(tmp_left) %*% tmp_left 
                      - (n_sample-wsum) / n_sample * t(tmp_right) %*% tmp_right)
  }
  gini
}

estimate_variance <- function(x, y, splits) {
  n_x <- nrow(x)
  n_class <- ncol(y)
  l1 <- x[, splits[1, 1]] < splits[1, 2]  ##left left
  r1 <- x[, splits[1, 1]] >= splits[1, 2] ##left right
  l2 <- x[, splits[2, 1]] < splits[2, 2]
  r2 <- x[, splits[2, 1]] >= splits[2, 2]
  
  d <- l1
  if (sum(d) > 1) {
    theta <- -apply(as.matrix(y[d, ]), c(2), sum) / sum(d)
  } else if (sum(d) == 1) {
    theta <- -y[d, ] 
  } else {
    theta <- rep(0, n_class)
  }
  
  d <- r1
  if (sum(d) > 1) {
    theta <- c(theta, -apply(as.matrix(y[d, ]), c(2), sum) / sum(d))
  } else if (sum(d) == 1) {
    theta <- c(theta, -y[d, ])
  } else {
    theta <- c(theta, rep(0, n_class))
  }
  
  d <- l2
  if (sum(d) > 1) {
    theta <- c(theta, apply(as.matrix(y[d, ]), c(2), sum) / sum(d))
  } else if (sum(d) == 1) {
    theta <- c(theta, y[d, ])
  } else {
    theta <- c(theta, rep(0, n_class))
  }
  
  d <- r2
  if (sum(d) > 1) {
    theta <- c(theta, apply(as.matrix(y[d, ]), c(2), sum) / sum(d))
  } else if (sum(d)==1) {
    theta <- c(theta, y[d, ])
  } else {
    theta <- c(theta, rep(0, n_class))
  }
  
  theta <- theta * 2
  big_y <- cbind(y * l1, y * r1, y * l2, y * r2)
  sigma <- cov(big_y)
  
  t(theta) %*% sigma %*% theta / n_x
}

compare_splits <- function(x, y, splits, max_row=10000) {  
  n_x <- nrow(x)
  gini <- get_gini(x, y, splits)
  order_gini <- order(gini, decreasing=FALSE)
  top <- order_gini[1]
  n_class <- ncol(y)
  n_splits <- nrow(splits)  
  
  alpha <- c()
  sigma <- c()
  for (i in 2:n_splits) {
    bot <- order_gini[i]
    delta <- gini[top] - gini[bot]
    if (n_x > max_row) {
      subsample_row <- sample(1:n_x, max_row) 
      sigma2hat <- estimate_variance(x[subsample_row, ], 
                                     y[subsample_row, ], 
                                     rbind(splits[top, ], splits[bot, ]))
    } else {
      sigma2hat <- estimate_variance(x, y, rbind(splits[top,], splits[bot,]))
    }
    # Numeric instability.
    if(sigma2hat<0) sigma2hat <- 0
    sigma <- c(sigma, sigma2hat)
    alpha <- c(alpha, pnorm(delta, mean=0, sd=sqrt(sigma2hat)))
  }
  list(order_gini=order_gini, 
       alpha=alpha, 
       sum_alpha=sum(alpha), 
       sigma=sigma, 
       gini=gini[order_gini])
}

test_splits <- function(splits, 
                        teacher,
                        generator, 
                        alpha=0.05, 
                        max_cut=10000, 
                        max_inc=1000, 
                        min_inc=10) {
  n_sample <- 1000
  alpha50 <- alpha / 50
  x <- generator(n_sample)
  y <- teacher(x)
  result <- compare_splits(x, y, splits)
  while (n_sample < max_cut && result$sum_alpha > alpha) {
    # Ditch some splits 
    tmp_splits <- splits[result$order_gini[1], ]
    for (i in 2:nrow(splits)) {
      if (result$alpha[i-1] > alpha50) {
        tmp_splits <- rbind(tmp_splits, splits[result$order_gini[i], ])
      }
    }
    splits <- tmp_splits
    alpha_top <- alpha / result$sum_alpha * result$alpha[1]
    thresh <- (qnorm(alpha_top) / qnorm(result$alpha[1]))^2
    inc_sample <- max(min(trunc(n_sample * (thresh-1)), max_inc), min_inc)

    new_x <- generator(inc_sample)
    new_y <- teacher(new_x)
    x <- rbind(x, new_x)
    y <- rbind(y, new_y)
    result <- compare_splits(x, y, splits)
    n_sample <- n_sample + inc_sample
  }
  
  if (result$sum_alpha > alpha) {
    # Test failed at given confidence. Alter to rule.
    suc <- FALSE
    label <- 1
    tmp_splits <- rbind(splits[result$order_gini[1], ])
    for (i in 2:length(result$order_gini)) {
      if (result$alpha[i-1] > alpha) {
        label <- label + 1
        tmp_splits <- rbind(tmp_splits, splits[result$order_gini[i], ])
      }
    }
    best <- which(tmp_splits[1:label, 1] == min(tmp_splits[1:label, 1]))
    best <- which(
      tmp_splits[1:label, 1] == min(tmp_splits[1:label, 1]) &
      tmp_splits[1:label, 2] == min(tmp_splits[best, 2]))
    best <- tmp_splits[best, ]
  } else {
    suc <- TRUE
    best <- splits[result$order_gini[1], ]
  }
  list(best=best, 
       success=suc, 
       n_sample=n_sample, 
       order_gini=result$order_gini, 
       alpha=result$alpha, 
       sigma=result$sigma, 
       gini=result$gini)
}

distillation_tree <- function(teacher, 
                              generator, 
                              split_num_per_var=5, 
                              split_num_cand=20,
                              split_digits=2,
                              splits=NULL,
                              confidence=0.05, 
                              stop_gini=1e-3,
                              stop_tree_depth=4,
                              max_sample_size=10000, 
                              max_stepsize=1000,
                              min_stepsize=10, 
                              # Below are internal to this function. 
                              # Don't specify when calling.
                              node_number=1, 
                              char="", 
                              upper=NULL,
                              lower=NULL) {
  alpha <- confidence
  tree_depth <- stop_tree_depth
  if (is.null(upper) || is.null(lower)) {
    data <- generator(NUM_EXAMPLES_FOR_SPLIT_DISCOVERY)
    n_cov <- ncol(data)
    lower <- rep(-INF, n_cov)
    upper <- rep(INF, n_cov)
  } else {
    data <- generator(NUM_EXAMPLES_FOR_SPLIT_DISCOVERY, lower, upper)
    n_cov <- ncol(data)
  }
  preds <- teacher(data)
  baseline_gini <- get_gini(data, preds, baseline=TRUE)
  
  # Find all potential candidate splits if not given
  if (is.null(splits)) {
    n_data <- nrow(data)
    splits <- NULL
    for (i in 1:n_cov) {
      sorted_cov <- sort(unique(signif(data[, i], digits=split_digits)))
      n_midpoints <- length(sorted_cov) - 1
      if (n_midpoints > split_num_per_var) {
        indices = floor(seq(from=1, 
                        to=n_midpoints, 
                        length.out=split_num_per_var)) 
      } else if (n_midpoints > 0) {
        indices <- 1:n_midpoints
      }
      if (n_midpoints > 0) {
        splits <- rbind(
          splits, cbind(rep(i, length(indices)),
                        (sorted_cov[indices] + sorted_cov[indices + 1]) / 2))
      }
    }
  }
  # Split the tree 
  if (node_number < 2^(stop_tree_depth - 1) && 
      baseline_gini > stop_gini && 
      !is.null(splits)) {
    gini <- get_gini(data, preds, splits)
    if (split_num_cand < nrow(splits)) {
      splits <- splits[order(gini)[1:split_num_cand], ]
    }
    # Find the best split, need to change the function accordingly
    bsplit <- test_splits(
      splits,
      teacher,
      function(n) generator(n, lower, upper),
      alpha=alpha, 
      max_cut=max_sample_size, 
      max_inc=max_stepsize, 
      min_inc=min_stepsize)
    # Zhengze's testing function
    # pvalues <- c()
    # for (TTime in 1:10) {
    #     testPoints <- ccg(xData, path)(10)
    #     pvalues <- c(pvalues, ptestu(testPoints, forest))
    # }
    # node$pvalues <- pvalues
    n_sample <- bsplit$n_sample
    bsplit <- bsplit$best
    node <- list()
    node$split <- c(bsplit, node_number, n_sample)
    node$leaf <- FALSE
    
    node$id <- paste(bsplit[1], round(bsplit[2], 2))
    tmp_upper <- upper
    tmp_upper[bsplit[1]] <- bsplit[2]
    tmp_lower <- lower
    tmp_lower[bsplit[1]] <- bsplit[2]
    node$lnode <- distillation_tree(
      teacher, 
      generator, 
      split_num_per_var=split_num_per_var, 
      split_num_cand=split_num_cand,
      split_digits=split_digits,
      splits=NULL,
      confidence=confidence, 
      stop_gini=stop_gini,
      stop_tree_depth=stop_tree_depth,
      max_sample_size=max_sample_size, 
      max_stepsize=max_stepsize,
      min_stepsize=min_stepsize, 
      node_number=node_number * 2, 
      char=paste(char, 'L', sep=""), 
      upper=tmp_upper,
      lower=lower)
    node$id <- paste(node$id, 'L', node$lnode$id)
    
    node$rnode <- distillation_tree(
      teacher, 
      generator, 
      split_num_per_var=split_num_per_var, 
      split_num_cand=split_num_cand,
      split_digits=split_digits,
      splits=NULL,
      confidence=confidence, 
      stop_gini=stop_gini,
      stop_tree_depth=stop_tree_depth,
      max_sample_size=max_sample_size, 
      max_stepsize=max_stepsize,
      min_stepsize=min_stepsize, 
      node_number=node_number * 2, 
      char=paste(char, 'L', sep=""), 
      upper=upper,
      lower=tmp_lower)
    node$id <- paste(node$id, 'R', node$rnode$id)
    node$splits <- as.matrix(rbind(node$split, node$lnode$splits, node$rnode$splits))
  } else {
    node <- list()
    node$leaf <- TRUE
    node$lnode <- NULL
    node$rnode <- NULL
    node$value <- apply(preds, c(2), mean)
  }
  node
}