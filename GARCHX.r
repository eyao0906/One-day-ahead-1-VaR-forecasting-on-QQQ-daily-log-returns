# =====================================================================
# QQQ VaR Forecasting: Dedicated GARCHX (VIX) Script
# =====================================================================

library(garchx)
library(readr)
library(dplyr)
library(ggplot2)

riskmetrics_ci <- function(mu, sigma, alpha, n_eff, confidence = 0.95) {
  a <- min(max(alpha, 1e-4), 0.2)
  q <- qnorm(a)
  f_q <- max(dnorm(q), 1e-8)
  se_q <- sqrt(a * (1 - a) / (max(n_eff, 1) * (f_q^2)))
  z <- qnorm(0.5 + confidence / 2)
  c(lower = mu + sigma * (q - z * se_q), upper = mu + sigma * (q + z * se_q))
}

kupiec_test <- function(violations, alpha) {
  v <- as.integer(violations)
  n <- length(v)
  x <- sum(v)

  pi_hat <- min(max(x / max(n, 1), 1e-8), 1 - 1e-8)
  a <- min(max(alpha, 1e-8), 1 - 1e-8)

  lr <- -2.0 * (
    (n - x) * log(1 - a) +
      x * log(a) -
      (n - x) * log(1 - pi_hat) -
      x * log(pi_hat)
  )
  pval <- 1 - pchisq(lr, df = 1)
  return(list(lr = lr, pval = pval))
}

christoffersen_test <- function(violations) {
  v <- as.integer(violations)
  if (length(v) < 2) return(list(lr = NA, pval = NA))

  n00 <- 0; n01 <- 0; n10 <- 0; n11 <- 0
  for (i in 2:length(v)) {
    prev <- v[i-1]; cur <- v[i]
    if (prev == 0 && cur == 0) n00 <- n00 + 1
    else if (prev == 0 && cur == 1) n01 <- n01 + 1
    else if (prev == 1 && cur == 0) n10 <- n10 + 1
    else n11 <- n11 + 1
  }

  p01 <- max(min(n01 / max(n00 + n01, 1), 1 - 1e-8), 1e-8)
  p11 <- max(min(n11 / max(n10 + n11, 1), 1 - 1e-8), 1e-8)
  p   <- max(min((n01 + n11) / max(n00 + n01 + n10 + n11, 1), 1 - 1e-8), 1e-8)

  ll_ind <- (n00 + n10) * log(1 - p) + (n01 + n11) * log(p)
  ll_dep <- n00 * log(1 - p01) + n01 * log(p01) + n10 * log(1 - p11) + n11 * log(p11)

  lr <- -2.0 * (ll_ind - ll_dep)
  pval <- 1 - pchisq(lr, df = 1)
  return(list(lr = lr, pval = pval))
}

main <- function() {
  data_path <- "data/model_data.csv"
  outdir <- "outputs/garchx_only"
  alpha <- 0.01
  initial_train_ratio <- 0.7

  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

  cat("Loading data...\n")
  df <- read_csv(data_path, show_col_types = FALSE) %>%
    select(Date, log_ret, vix_close) %>%
    na.omit() %>%
    arrange(Date)

  df$log_ret_scaled <- df$log_ret * 100
  df$vix_scaled <- as.numeric(scale(df$vix_close))

  n <- nrow(df)
  train_end0 <- floor(n * initial_train_ratio)
  if (train_end0 < 500) train_end0 <- min(max(500, train_end0), n - 30)

  results <- list()
  q_norm <- qnorm(alpha)

  cat(sprintf("Starting rolling forecast. Evaluating %d days...\n", n - train_end0))

  for (t in train_end0:(n - 1)) {
    train_data <- df[1:t, ]
    test_row <- df[t + 1, ]

    r_train <- train_data$log_ret_scaled
    vix_train <- matrix(train_data$vix_scaled, ncol = 1)

    mu <- mean(r_train)
    y_adj <- r_train - mu

    var_garchx <- NA
    lower_bound <- NA
    upper_bound <- NA
    sigma_next_x <- NA
    tryCatch({
      fit_garchx <- garchx(y_adj, order = c(1, 1), xreg = vix_train)
      vix_T <- matrix(tail(vix_train, 1), ncol = 1)
      pred_garchx <- predict(fit_garchx, n.ahead = 1, newxreg = vix_T)
      sigma_next_x <- sqrt(as.numeric(pred_garchx))

      var_garchx <- (mu + sigma_next_x * q_norm) / 100
      ci <- riskmetrics_ci(mu = mu / 100, sigma = sigma_next_x / 100, alpha = alpha, n_eff = length(y_adj))
      lower_bound <- unname(ci["lower"])
      upper_bound <- unname(ci["upper"])
    }, error = function(e) {
      cat("Warning: Optimization failed at step", t, "\n")
    })

    realized <- test_row$log_ret
    date <- test_row$Date

    if (!is.na(var_garchx)) {
      results[[length(results) + 1]] <- data.frame(
        Date = date,
        Model = "GARCHX(VIX) [garchx pkg]",
        VaR = var_garchx,
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        Return = realized,
        Violation = ifelse(realized < var_garchx, 1, 0),
        meta_mu = mu / 100,
        meta_sigma = sigma_next_x / 100
      )
    }

    if (t %% 100 == 0) cat(sprintf("Processed step %d / %d\n", t, n - 1))
  }

  forecasts <- bind_rows(results)
  write_csv(forecasts, file.path(outdir, "var_garchx_forecasts.csv"))

  cat("\nCalculating Backtest Statistics...\n")
  v <- forecasts$Violation

  k_test <- kupiec_test(v, alpha)
  c_test <- christoffersen_test(v)
  lr_cc <- k_test$lr + c_test$lr
  p_cc <- 1 - pchisq(lr_cc, df = 2)

  summary_df <- data.frame(
    Model = "GARCHX(VIX) [garchx pkg]",
    N = length(v),
    Violations = sum(v),
    ExpectedViolations = length(v) * alpha,
    ViolationRate = mean(v),
    Kupiec_LRuc = k_test$lr,
    Kupiec_pvalue = k_test$pval,
    Christoffersen_LRind = c_test$lr,
    Christoffersen_pvalue = c_test$pval,
    ConditionalCoverage_LRcc = lr_cc,
    ConditionalCoverage_pvalue = p_cc
  )

  write_csv(summary_df, file.path(outdir, "var_garchx_backtests.csv"))

  cat("\n--- GARCHX BACKTEST SUMMARY ---\n")
  print(summary_df)

  cat("\nGenerating Plot...\n")
  viol_df <- forecasts %>% filter(Violation == 1)

  p <- ggplot(forecasts, aes(x = Date)) +
    geom_line(aes(y = Return, color = "Realized Return"), linewidth = 0.5, alpha = 0.7) +
    geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound, fill = "95% CI"), alpha = 0.15, colour = NA) +
    geom_line(aes(y = VaR, color = "1-Day VaR"), linewidth = 0.8) +
    scale_color_manual(name = "", values = c("Realized Return" = "black", "1-Day VaR" = "red")) +
    scale_fill_manual(name = "", values = c("95% CI" = "red")) +
    theme_minimal() +
    labs(title = "VaR Backtest: GARCHX with VIX Overlay", y = "Daily Log Return", x = "") +
    theme(legend.position = "bottom")

  if (nrow(viol_df) > 0) {
    p <- p + geom_point(data = viol_df, aes(y = Return, shape = "Violation"), color = "blue", size = 2) +
      scale_shape_manual(name = "", values = c("Violation" = 16))
  }

  ggsave(file.path(outdir, "plot_garchx_vix.png"), plot = p, width = 10, height = 4, dpi = 150)
  cat(sprintf("Forecasts, summary, and plot saved to: %s\n", outdir))
}

main()
