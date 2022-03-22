# set up
library(tidyverse)
library(cowplot)

# Simulate ramp data
roll <- function(x, n) {
  x[(seq_along(x) - (n+1)) %% length(x) + 1]
}

n_shuffles <- 100

# Choose a random number to shift the ramps buy
random_start <- runif(1, 0, 100)

# Make ramps and randomly shifted ramps
single_ramps <- tibble(pos = c(1:100),
                       single_ramp = c(1:50,1:50),
                       single_ramp_shift = roll(c(1:50, 1:50), round(random_start))) %>%
  gather("ramp_type", "rate", -pos)


# Plot
ggplot(single_ramps, aes(pos, rate, colour = ramp_type)) +
  geom_point()


# Make a concatenated version of each ramp
multi_ramps <- map_dfr(seq_len(n_shuffles), ~single_ramps)

# Plot (trivially this looks the same as the single ramp plot)
ggplot(multi_ramps, aes(pos, rate, colour = ramp_type)) +
  geom_point()

# Function to introduce jitter between consecutive ramps
jitter_singles <- function(jit = 10, random_start = 25, shuffle_type = "continuous") {
  random_jitter = round(random_start)

  out_df <- tibble(pos = c(1:100),
                   multi_ramp = c(1:50,1:50),
                   multi_ramp_shift = roll(c(1:50, 1:50), random_jitter)) %>%
    gather("ramp_type", "rate", -pos)
  
  for (i in 2:n_shuffles) {
    random_jitter <- ifelse(identical(shuffle_type,"continuous"),
                            random_jitter + round(runif(1, -jit/2, jit/2)),
                            random_start + round(runif(1, 0, jit)))
    another_jitter <- tibble(pos = c(1:100),
                             multi_ramp = c(1:50,1:50),
                             multi_ramp_shift = roll(c(1:50, 1:50), random_jitter)) %>%
      gather("ramp_type", "rate", -pos)
    out_df <- rbind(out_df, another_jitter)
  }
  out_df
}

# To plot jittered data with a mean overlaid
plot_multi <- function(df) {
  m_r_j_mean <- df %>%
    filter(ramp_type == "multi_ramp_shift") %>%
    group_by(pos) %>%
    summarise(rate = mean(rate))
  ggplot(df, aes(pos, rate, colour = ramp_type)) +
    geom_line() +
    geom_line(data = m_r_j_mean, (aes(pos, rate)), colour = "black")
}

# No jitter, circular shuffle
multi_ramps_jitter_low <- jitter_singles(1, 25)
multi_ramps_jitter_low_plot <- plot_multi(multi_ramps_jitter_low)

# Moderate jitter, circular shuffle
multi_ramps_jitter_moderate <- jitter_singles(10, 25)
multi_ramps_jitter_moderate_plot <- plot_multi(multi_ramps_jitter_moderate)


# High jitter, circular shuffle
multi_ramps_jitter_high <- jitter_singles(25, 25)
multi_ramps_jitter_high_plot <- plot_multi(multi_ramps_jitter_high)

# Resetting shuffle.
# With jitter = 100 each shuffle has equal probability of being from anywhere on the track.
multi_ramps_reset <- jitter_singles(jit = 100, random_start = 25, shuffle_type = "reset")
multi_ramps_reset_plot <- plot_multi(multi_ramps_reset)


plot_grid(multi_ramps_jitter_low_plot,
          multi_ramps_jitter_moderate_plot,
          multi_ramps_jitter_high_plot,
          multi_ramps_reset_plot)

ggsave("plots/shuffle_comparisons.jpg")
