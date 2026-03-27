# =========================
# QQQ VaR Project - One-file R pipeline
# Replaces:
#   - var_utils.py
#   - run_var_project.py
#   - plot_var_results.py
# =========================
required_pkgs <- c("rugarch", "quantreg")
missing_pkgs <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    paste(
      "Missing required packages:",
      paste(missing_pkgs, collapse = ", "),
      "\nInstall with install.packages(c(",
      paste(sprintf('"%s"', missing_pkgs), collapse = ", "),
      "))"
    )
  )
}

library(rugarch)
library(quantreg)

# -------------------------
# User settings
# -------------------------
data_path <- "C:/Users/Ethan Yao/.openclaw/workspace/timeseries_variance_project/data/model_data.csv"      # change if needed
outdir <- "outputs_r"
plotdir <- file.path(outdir, "plots")
alpha <- 0.01
initial_train_frac <- 0.70
hs_window <- 250
max_steps <- NA_integer_           # set to integer if you want a shorter run

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
dir.create(plotdir, showWarnings = FALSE, recursive = TRUE)

# -------------------------
# Helpers
# -------------------------
safe_alpha <- function(a) {
  max(min(a, 0.20), 1e-4)
}

prepare_garchx_regressor <- function(x) {
  x <- pmax(as.numeric(x), 1e-6)
  z <- log(x)
  z <- z / mean(z, na.rm = TRUE)
  z <- pmin(pmax(z, 0.25), 4.0)
  z
}

forecast_hs <- function(returns, alpha = 0.01, window = 250) {
  r <- as.numeric(returns)
  window <- min(window, length(r))
  q <- as.numeric(quantile(r[(length(r) - window + 1):length(r)], probs = safe_alpha(alpha), type = 7, na.rm = TRUE))
  list(VaR = q, meta = list(window = window))
}

forecast_garch <- function(returns, alpha = 0.01, dist = "std") {
  r <- as.numeric(returns) * 100
  spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = dist
  )
  fit <- ugarchfit(spec = spec, data = r, solver = "hybrid", solver.control = list(trace = 0))
  fc <- ugarchforecast(fit, n.ahead = 1)

  mu <- as.numeric(fitted(fc))[1] / 100
  sigma <- as.numeric(sigma(fc))[1] / 100

  coefs <- coef(fit)
  shape <- if ("shape" %in% names(coefs)) as.numeric(coefs["shape"]) else NA_real_

  if (dist == "std" && is.finite(shape)) {
    q <- rugarch::qdist("std", p = safe_alpha(alpha), mu = 0, sigma = 1, shape = shape)
  } else {
    q <- qnorm(safe_alpha(alpha))
  }

  list(
    VaR = mu + sigma * q,
    meta = list(mu = mu, sigma = sigma, shape = shape)
  )
}

forecast_gjr <- function(returns, alpha = 0.01, dist = "std") {
  r <- as.numeric(returns) * 100
  spec <- ugarchspec(
    variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = dist
  )
  fit <- ugarchfit(spec = spec, data = r, solver = "hybrid", solver.control = list(trace = 0))
  fc <- ugarchforecast(fit, n.ahead = 1)

  mu <- as.numeric(fitted(fc))[1] / 100
  sigma <- as.numeric(sigma(fc))[1] / 100

  coefs <- coef(fit)
  shape <- if ("shape" %in% names(coefs)) as.numeric(coefs["shape"]) else NA_real_

  if (dist == "std" && is.finite(shape)) {
    q <- rugarch::qdist("std", p = safe_alpha(alpha), mu = 0, sigma = 1, shape = shape)
  } else {
    q <- qnorm(safe_alpha(alpha))
  }

  list(
    VaR = mu + sigma * q,
    meta = list(mu = mu, sigma = sigma, shape = shape)
  )
}

forecast_garchx <- function(returns, x, alpha = 0.01, dist = "std", fallback_to_garch = TRUE) {
  r <- as.numeric(returns) * 100
  xreg <- matrix(prepare_garchx_regressor(x), ncol = 1)

  spec <- ugarchspec(
    variance.model = list(
      model = "sGARCH",
      garchOrder = c(1, 1),
      external.regressors = xreg
    ),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = dist
  )

  fit <- tryCatch(
    ugarchfit(spec = spec, data = r, solver = "hybrid", solver.control = list(trace = 0)),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    if (fallback_to_garch) {
      out <- forecast_garch(returns, alpha = alpha, dist = dist)
      out$meta$fallback <- 1
      return(out)
    } else {
      stop("GARCHX fit failed")
    }
  }

  xnext <- matrix(tail(xreg, 1), ncol = 1)
  fc <- tryCatch(
    ugarchforecast(fit, n.ahead = 1, external.forecasts = list(vregfor = xnext)),
    error = function(e) NULL
  )

  if (is.null(fc)) {
    if (fallback_to_garch) {
      out <- forecast_garch(returns, alpha = alpha, dist = dist)
      out$meta$fallback <- 1
      return(out)
    } else {
      stop("GARCHX forecast failed")
    }
  }

  mu <- as.numeric(fitted(fc))[1] / 100
  sigma <- as.numeric(sigma(fc))[1] / 100

  coefs <- coef(fit)
  shape <- if ("shape" %in% names(coefs)) as.numeric(coefs["shape"]) else NA_real_

  if (dist == "std" && is.finite(shape)) {
    q <- rugarch::qdist("std", p = safe_alpha(alpha), mu = 0, sigma = 1, shape = shape)
  } else {
    q <- qnorm(safe_alpha(alpha))
  }

  list(
    VaR = mu + sigma * q,
    meta = list(mu = mu, sigma = sigma, shape = shape, fallback = 0)
  )
}

# CAViaR-SAV:
# q_t = b0 + b1 q_{t-1} + b2 |r_{t-1}|
caviar_sav_path <- function(params, r, alpha) {
  b0 <- params[1]
  b1 <- params[2]
  b2 <- params[3]
  q <- numeric(length(r))
  q[1] <- as.numeric(quantile(r, probs = alpha, type = 7, na.rm = TRUE))
  if (length(r) >= 2) {
    for (i in 2:length(r)) {
      q[i] <- b0 + b1 * q[i - 1] + b2 * abs(r[i - 1])
    }
  }
  q
}

caviar_loss <- function(params, r, alpha) {
  q <- caviar_sav_path(params, r, alpha)
  u <- r - q
  sum(u * (alpha - as.numeric(u < 0)))
}

forecast_caviar_sav <- function(returns, alpha = 0.01) {
  r <- as.numeric(returns)

  starts <- list(
    c(-0.0010, 0.80, 0.10),
    c(-0.0005, 0.90, 0.05),
    c(-0.0015, 0.70, 0.20)
  )

  best <- NULL
  best_val <- Inf

  for (s in starts) {
    fit <- tryCatch(
      optim(
        par = s,
        fn = caviar_loss,
        r = r,
        alpha = alpha,
        method = "L-BFGS-B",
        lower = c(-0.05, -0.50, 0.00),
        upper = c( 0.05,  1.50, 2.00),
        control = list(maxit = 2000)
      ),
      error = function(e) NULL
    )

    if (!is.null(fit) && is.finite(fit$value) && fit$value < best_val) {
      best <- fit
      best_val <- fit$value
    }
  }

  if (is.null(best)) stop("CAViaR optimization failed")

  params <- best$par
  q_path <- caviar_sav_path(params, r, alpha)
  q_next <- params[1] + params[2] * tail(q_path, 1) + params[3] * abs(tail(r, 1))

  list(
    VaR = as.numeric(q_next),
    meta = list(b0 = params[1], b1 = params[2], b2 = params[3])
  )
}

kupiec_test <- function(violations, alpha) {
  v <- as.integer(violations)
  n <- length(v)
  x <- sum(v)
  pi_hat <- min(max(x / max(n, 1), 1e-8), 1 - 1e-8)
  a <- min(max(alpha, 1e-8), 1 - 1e-8)

  lr <- -2 * (
    (n - x) * log(1 - a) +
      x * log(a) -
      (n - x) * log(1 - pi_hat) -
      x * log(pi_hat)
  )

  pval <- 1 - pchisq(lr, df = 1)
  c(stat = as.numeric(lr), pvalue = as.numeric(pval))
}

christoffersen_independence_test <- function(violations) {
  v <- as.integer(violations)
  if (length(v) < 2) return(c(stat = NA_real_, pvalue = NA_real_))

  n00 <- 0; n01 <- 0; n10 <- 0; n11 <- 0
  for (i in 2:length(v)) {
    prev <- v[i - 1]
    cur <- v[i]
    if (prev == 0 && cur == 0) n00 <- n00 + 1
    if (prev == 0 && cur == 1) n01 <- n01 + 1
    if (prev == 1 && cur == 0) n10 <- n10 + 1
    if (prev == 1 && cur == 1) n11 <- n11 + 1
  }

  p01 <- n01 / max(n00 + n01, 1)
  p11 <- n11 / max(n10 + n11, 1)
  p <- (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

  p01 <- min(max(p01, 1e-8), 1 - 1e-8)
  p11 <- min(max(p11, 1e-8), 1 - 1e-8)
  p <- min(max(p, 1e-8), 1 - 1e-8)

  ll_ind <- (n00 + n10) * log(1 - p) + (n01 + n11) * log(p)
  ll_dep <- n00 * log(1 - p01) + n01 * log(p01) + n10 * log(1 - p11) + n11 * log(p11)

  lr <- -2 * (ll_ind - ll_dep)
  pval <- 1 - pchisq(lr, df = 1)
  c(stat = as.numeric(lr), pvalue = as.numeric(pval))
}

conditional_coverage_test <- function(violations, alpha) {
  uc <- kupiec_test(violations, alpha)
  ind <- christoffersen_independence_test(violations)
  lr_cc <- uc["stat"] + ind["stat"]
  pval <- 1 - pchisq(lr_cc, df = 2)
  c(stat = as.numeric(lr_cc), pvalue = as.numeric(pval))
}

# -------------------------
# Load data
# -------------------------
df <- read.csv(data_path, stringsAsFactors = FALSE)
if (!all(c("Date", "log_ret", "vix_close") %in% names(df))) {
  stop("Input data must contain columns: Date, log_ret, vix_close")
}

df$Date <- as.Date(df$Date)
df <- df[order(df$Date), c("Date", "log_ret", "vix_close")]
df <- df[complete.cases(df), ]

n <- nrow(df)
train_end0 <- floor(n * initial_train_frac)
if (train_end0 < 500) train_end0 <- min(max(500, train_end0), n - 30)

end_limit <- n
if (!is.na(max_steps)) {
  end_limit <- min(n, train_end0 + max_steps)
}

# -------------------------
# Rolling forecasts
# -------------------------
rows <- list()
row_id <- 1L

for (t in train_end0:(end_limit - 1)) {
  train <- df[1:t, ]
  test_row <- df[t + 1, ]

  r_train <- train$log_ret
  x_train <- train$vix_close
  realized <- as.numeric(test_row$log_ret)
  date <- test_row$Date

  model_list <- list(
    "GARCH(1,1)" = function() forecast_garch(r_train, alpha = alpha, dist = "std"),
    "GARCHX(VIX)" = function() forecast_garchx(r_train, x_train, alpha = alpha, dist = "std", fallback_to_garch = TRUE),
    "GJR-GARCH" = function() forecast_gjr(r_train, alpha = alpha, dist = "std"),
    "CAViaR-SAV" = function() forecast_caviar_sav(r_train, alpha = alpha),
    "Historical-Simulation" = function() forecast_hs(r_train, alpha = alpha, window = hs_window)
  )

  for (model_name in names(model_list)) {
    out <- tryCatch(model_list[[model_name]](), error = function(e) e)

    if (inherits(out, "error")) {
      message(sprintf("[warn] step %s, model %s failed: %s", t, model_name, out$message))
      next
    }

    hit <- as.integer(realized < out$VaR)

    row <- data.frame(
      Date = date,
      Model = model_name,
      alpha = alpha,
      VaR = as.numeric(out$VaR),
      Return = realized,
      Violation = hit,
      stringsAsFactors = FALSE
    )

    if (!is.null(out$meta) && length(out$meta) > 0) {
      for (nm in names(out$meta)) {
        val <- out$meta[[nm]]
        row[[paste0("meta_", nm)]] <- ifelse(is.finite(as.numeric(val)), as.numeric(val), NA_real_)
      }
    }

    rows[[row_id]] <- row
    row_id <- row_id + 1L
  }
}

bind_rows_fill <- function(lst) {
  all_names <- unique(unlist(lapply(lst, names)))
  aligned <- lapply(lst, function(df) {
    missing <- setdiff(all_names, names(df))
    for (nm in missing) df[[nm]] <- NA
    df <- df[, all_names, drop = FALSE]
    df
  })
  do.call(rbind, aligned)
}
forecasts <- bind_rows_fill(rows)
write.csv(forecasts, file.path(outdir, "var_forecasts_r.csv"), row.names = FALSE)

# -------------------------
# Backtest summary
# -------------------------
models <- unique(forecasts$Model)
summary_rows <- list()

for (i in seq_along(models)) {
  m <- models[i]
  g <- forecasts[forecasts$Model == m, ]
  v <- g$Violation

  uc <- kupiec_test(v, alpha)
  ind <- christoffersen_independence_test(v)
  cc <- conditional_coverage_test(v, alpha)

  summary_rows[[i]] <- data.frame(
    Model = m,
    N = nrow(g),
    Violations = sum(v),
    ExpectedViolations = alpha * nrow(g),
    ViolationRate = mean(v),
    Kupiec_LRuc = uc["stat"],
    Kupiec_pvalue = uc["pvalue"],
    Christoffersen_LRind = ind["stat"],
    Christoffersen_pvalue = ind["pvalue"],
    ConditionalCoverage_LRcc = cc["stat"],
    ConditionalCoverage_pvalue = cc["pvalue"],
    stringsAsFactors = FALSE
  )
}

backtests <- do.call(rbind, summary_rows)
backtests <- backtests[order(-backtests$ConditionalCoverage_pvalue), ]
write.csv(backtests, file.path(outdir, "var_backtests_r.csv"), row.names = FALSE)

print(backtests)

# -------------------------
# Plots
# -------------------------
for (m in unique(forecasts$Model)) {
  g <- forecasts[forecasts$Model == m, ]
  g <- g[order(g$Date), ]

  png(
    filename = file.path(plotdir, paste0("var_", gsub("[/() ]", "_", m), ".png")),
    width = 1400,
    height = 500,
    res = 140
  )

  plot(
    g$Date, g$Return,
    type = "l",
    lwd = 1,
    xlab = "Date",
    ylab = "Return / VaR",
    main = paste("VaR Backtest:", m)
  )
  lines(g$Date, g$VaR, lwd = 1)
  viol <- g[g$Violation == 1, ]
  if (nrow(viol) > 0) {
    points(viol$Date, viol$Return, pch = 16, cex = 0.7)
  }
  legend(
    "bottomleft",
    legend = c("Realized Return", "Forecast VaR", "Violations"),
    lty = c(1, 1, NA),
    pch = c(NA, NA, 16),
    bty = "n"
  )
  dev.off()
}

cat("\nSaved:\n")
cat(file.path(outdir, "var_forecasts_r.csv"), "\n")
cat(file.path(outdir, "var_backtests_r.csv"), "\n")
cat(plotdir, "\n")