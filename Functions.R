### This script contains functions which are called in the Analysis markdown code


# Adds positions to a single column data frame that contains a neurons binned mean firing rate
add_position <- function(df, session_id = "", cluster_id = "") {
  len = length(unlist(df))
  df <- tibble(Rates = unlist(df), Position = rep(1:len)) 
  #if(all(is.na(df$Rates))){print(paste0("All NAs. Session: ", session_id, ". Cluster:", cluster_id))}
  df
}


## ----------------------------------------------------------##

lm_helper <- function(df, bins){
  if(all(is.na(df))) 
    return(NA)
  df_mod <- lm(Rates ~ Position, data = df[bins,], na.action=na.omit)
}


# LM helper function for plotting speed against spike rate
speed_lm_helper <- function(df, bins){
  df_mod <- lm(SpikeRates ~ SpeedRates, data = df[bins,])
}


# LM helper function for plotting time against spike rate
time_lm_helper <- function(df, bins){
  if(all(is.na(df))) 
    return(NA)
  df_mod <- lm(SpikeRates ~ TimeRates, data = df[bins,])
}


lm_helper_with_trialtype <- function(df, bins){
  if(all(is.na(df))) 
    return(NA)
  df_mod <- lmer(Rates ~ Position + (1|Type), data = df[bins,])
}



## ----------------------------------------------------------##

## this function averages spike rate over trials
# input: spike_rate_on_trials
#output: tibble of average rates at positions

average_over_trials <- function(df, trial_indicator=2) {
  df <- tibble(SpikeNum = df[[1]], Trials = df[[2]], Types = df[[3]], Position = c(rep(1:200, times=nrow(unique(df[[2]])))))
  df %>% 
    filter(Types == trial_indicator) %>% 
    select(SpikeNum, Position, Trials) %>% 
    group_by(Position) %>% 
    summarise(Rates = n(), Rates = mean(SpikeNum), Rates_error = sd(SpikeNum))
}

average_speed_over_trials <- function(df, trial_indicator=2) {
  df <- tibble(Spikenum = df[[1]], Speed = df[[2]], Trials = df[[4]], Types = df[[5]], Position = c(rep(1:200, times=max(df[[4]]))))
  df %>% 
    filter(Types == trial_indicator) %>% 
    select(Spikenum, Speed, Position, Trials) %>% 
    group_by(Position) %>% 
    summarise(SpikeRates = n(), SpikeRates = mean(Spikenum), SpeedRates = n(), SpeedRates = mean(Speed, na.rm=TRUE))
}


average_error_over_trials <- function(df, trial_indicator=2) {
  df <- tibble(SpikeNum = df[[1]], Trials = df[[2]], Types = df[[3]], Position = c(rep(1:200, times=nrow(unique(df[[2]])))))
  df %>% 
    filter(Types == trial_indicator) %>% 
    select(SpikeNum, Position, Trials) %>% 
    group_by(Position) %>% 
    summarise(Rates = n(), Rates = mean(SpikeNum))
}

average_over_trials_in_time <- function(df, trial_indicator=2) {
  df <- tibble(SpikeNum = as.numeric(df[[1]]), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df %>% 
    filter(Types == trial_indicator) %>% 
    select(SpikeNum, Position, Trials) %>% 
    group_by(Position) %>% 
    summarise(Rates = n(), Rates = mean(SpikeNum), Rates_error = sd(SpikeNum))
}

## ----------------------------------------------------------##



#Next we want to perform LM on each cluster to analyse firing rate verses location.
#- Make a new columns in dataframe for output of linear model
#- Fit the model from average rates versus location for each cluster
#- Then put the results into new columns of the dataframe

lm_analysis <- function(df, spike_rate_col, startbin = 30, endbin = 170) {
  spike_rate_col <- enquo(spike_rate_col)
  out_name <- sym(paste0(quo_name(spike_rate_col)))
  sr_unnest_name <- sym(paste0(quo_name(spike_rate_col), "_unnest"))
  fit_name <- sym(paste0(quo_name(out_name), "_fit"))
  glance_name <- sym(paste0(quo_name(out_name), "_glance"))
  r2_name <- sym(paste0(quo_name(out_name), "_r2"))
  Pval_name <- sym(paste0(quo_name(out_name), "_Pval"))
  slope_name <- sym(paste0(quo_name(out_name), "_slope"))
  intercept_name <- sym(paste0(quo_name(out_name), "_intercept"))
  df <- df %>%
    mutate(!!fit_name := map(!!spike_rate_col, bins=c(startbin:endbin), lm_helper),
           !!glance_name := map(!!fit_name, glance),
           !!r2_name := map_dbl(!!glance_name, ~.$r.squared),
           !!Pval_name := map_dbl(!!glance_name, ~.$p.value),
           !!slope_name := map_dbl(!!fit_name, ~.$coefficients[2]),
           !!intercept_name := map_dbl(!!fit_name, ~.$coefficients[1]))
}


#Next we want to perform LM on each cluster to analyse firing rate verses speed and time
#- Make a new columns in dataframe for output of linear model
#- Fit the model from average rates versus location for each cluster
#- Then put the results into new columns of the dataframe

speed_lm_analysis <- function(df, spike_rate_col, startbin = 30, endbin = 170) {
  spike_rate_col <- enquo(spike_rate_col)
  out_name <- sym(paste0(quo_name(spike_rate_col)))
  sr_unnest_name <- sym(paste0(quo_name(spike_rate_col), "_unnest"))
  fit_name <- sym(paste0(quo_name(out_name), "_fit"))
  glance_name <- sym(paste0(quo_name(out_name), "_glance"))
  r2_name <- sym(paste0(quo_name(out_name), "_r2"))
  Pval_name <- sym(paste0(quo_name(out_name), "_Pval"))
  slope_name <- sym(paste0(quo_name(out_name), "_slope"))
  df <- df %>%
    mutate(!!fit_name := map(!!spike_rate_col, bins=c(startbin:endbin), speed_lm_helper),
           !!glance_name := map(!!fit_name, glance),
           !!r2_name := map_dbl(!!glance_name, ~.$r.squared),
           !!Pval_name := map_dbl(!!glance_name, ~.$p.value),
           !!slope_name := map_dbl(!!fit_name, ~.$coefficients[2]))
}



## same as above but for time
time_lm_analysis <- function(df, spike_rate_col, startbin = 30, endbin = 170) {
  spike_rate_col <- enquo(spike_rate_col)
  out_name <- sym(paste0(quo_name(spike_rate_col)))
  sr_unnest_name <- sym(paste0(quo_name(spike_rate_col), "_unnest"))
  fit_name <- sym(paste0(quo_name(out_name), "_fit"))
  glance_name <- sym(paste0(quo_name(out_name), "_glance"))
  r2_name <- sym(paste0(quo_name(out_name), "_r2"))
  Pval_name <- sym(paste0(quo_name(out_name), "_Pval"))
  slope_name <- sym(paste0(quo_name(out_name), "_slope"))
  df <- df %>%
    mutate(!!fit_name := map(!!spike_rate_col, bins=c(startbin:endbin), time_lm_helper),
           !!glance_name := map(!!fit_name, glance),
           !!r2_name := map_dbl(!!glance_name, ~.$r.squared),
           !!Pval_name := map_dbl(!!glance_name, ~.$p.value),
           !!slope_name := map_dbl(!!fit_name, ~.$coefficients[2]))
}




lm_analysis_named <- function(df, spike_rate_col, startbin = 30, endbin = 170) {
  spike_rate_col <- enquo(spike_rate_col)
  out_name <- sym(paste0(quo_name(spike_rate_col), startbin, "_", endbin))
  sr_unnest_name <- sym(paste0(quo_name(out_name), "_unnest"))
  fit_name <- sym(paste0(quo_name(out_name), "_fit"))
  glance_name <- sym(paste0(quo_name(out_name), "_glance"))
  r2_name <- sym(paste0(quo_name(out_name), "_r2"))
  slope_name <- sym(paste0(quo_name(out_name), "_slope"))
  intercept_name <- sym(paste0(quo_name(out_name), "_intercept"))
  df <- df %>%
    mutate(!!fit_name := map(!!spike_rate_col, bins=c(startbin:endbin), lm_helper),
           !!glance_name := map(!!fit_name, glance),
           !!r2_name := map_dbl(!!glance_name, ~.$r.squared),
           !!slope_name := map_dbl(!!fit_name, ~.$coefficients[2]),
           !!intercept_name := map_dbl(!!fit_name, ~.$coefficients[1]))
}


## ----------------------------------------------------------##


# make function which returns results to plot in lm
#1. function 1 : returns tibble with trial type == beaconed, non-beaconed & probe
#2. function 2 : returns tibble with trial type == shuffled & real

return_lm_results_trial_types <- function(df){
  tibble(
    Trial_type = factor(c(rep(c("Beaconed","Non-beaconed", "Probe"), each =nrow(df)))),
    Slopes = c(df$asr_b_slope, df$asr_nb_slope, df$asr_p_slope), 
    r2 = c(df$asr_b_r2, df$asr_nb_r2, df$asr_p_r2), 
    id = c(df$cluster_id, df$cluster_id, df$cluster_id))
}

## ----------------------------------------------------------##

# needs to be taken into one function

#function2 - shuffled and real : beaconed trials
return_b_shuffle_lm_results <- function(df){
  tibble(
    Trial_type = factor(c(rep(c("Real","Shuffled"), each =nrow(df)))),
    Slopes = c(df$asr_b_slope, df$shuff_asr_b_slope), 
    r2 = c(df$asr_b_r2, df$shuff_asr_b_r2), 
    id = c(df$cluster_id, df$cluster_id))
}



#function2 - shuffled and real : beaconed trials
return_p_shuffle_lm_results <- function(df){
  tibble(
    Trial_type = factor(c(rep(c("Real","Shuffled"), each =nrow(df)))),
    Slopes = c(df$asr_p_slope, df$shuff_asr_p_slope), 
    r2 = c(df$asr_p_r2, df$shuff_asr_p_r2), 
    id = c(df$cluster_id, df$cluster_id))
}


## ----------------------------------------------------------##


# scatter plot of lm results


lm_plot <- function(lm_results){
  ggplot(lm_results, aes(x = Slopes, y = r2, color=Trial_type)) + 
    geom_point() +
    xlab("\nslope") +
    ylab("R2") +
    theme_classic() +
    theme(axis.text.x = element_text(size=16),
          axis.text.y = element_text(size=16),
          legend.position="bottom", 
          legend.title = element_blank(),
          text = element_text(size=16), 
          legend.text=element_text(size=16), 
          axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))) 
  }

## ----------------------------------------------------------##

normalit<-function(m){
  (m - min(m))/(max(m)-min(m))
}

## ----------------------------------------------------------##


scale_firing_rate <- function(df){
  Rates1 <- scale(df$Rates)
  Rates1[is.na(Rates1)]<-0
  df <- tibble(Position = df$Position, Rates = Rates1)
}


## ----------------------------------------------------------##


lm_analysis_2 <- function(df, spike_rate_col, startbin = 30, endbin = 170, trackseg="start") {
  spike_rate_col <- enquo(spike_rate_col)
  out_name <- sym(paste0(quo_name(spike_rate_col), "_", trackseg))
  sr_unnest_name <- sym(paste0(quo_name(spike_rate_col), "_unnest"))
  fit_name <- sym(paste0(quo_name(out_name), "_fit"))
  glance_name <- sym(paste0(quo_name(out_name), "_glance"))
  r2_name <- sym(paste0(quo_name(out_name), "_r2"))
  slope_name <- sym(paste0(quo_name(out_name), "_slope"))
  df <- df %>%
    mutate(!!fit_name := map(!!spike_rate_col, bins=c(startbin:endbin), lm_helper),
           !!glance_name := map(!!fit_name, glance),
           !!r2_name := map_dbl(!!glance_name, ~.$r.squared),
           !!slope_name := map_dbl(!!fit_name, ~.$coefficients[2]))
}


## ----------------------------------------------------------##


heatmap_plot <- function(df){ 
  ggplot(df, aes(Position, new_cluster_id)) +
    labs(x = "\nPosition (cm)", y = "slope number") +
    #geom_smooth(method ="lm") +
    coord_cartesian() +
    scale_color_gradient() +
    geom_tile(aes(fill = asr_b)) +
    scale_fill_distiller(palette = "Spectral") +
    labs(fill="Norm firing rate") +
    theme_classic() +
    theme(axis.text.x = element_text(size=14),
          axis.text.y = element_text(size=14),
          legend.position="bottom", 
          legend.title = element_blank(),
          text = element_text(size=14), 
          legend.text=element_text(size=14), 
          axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))
  }


## ----------------------------------------------------------##


overlaid_rates_plot <- function(df){ 
  ggplot(df, aes(x=Position, y=asr_b_smooth)) +
    labs(x = "\nPosition (cm)", y = "Firing rate (Hz)", color = "Cluster id") +
    geom_smooth(aes(x=Position, y=asr_b_smooth, group=new_cluster_id), size=0.4, weight=0.5, alpha=0.5, colour="grey71", span = 0.2, method = 'loess', se=FALSE) +
    geom_smooth(aes(x=Position, y=asr_b_smooth), size=0.8, weight=1, alpha=0.5, colour="red2", span = 0.2, method = 'loess', se=TRUE) +
    guides(fill=FALSE) +
    theme_classic() +
    theme(axis.text.x = element_text(size=14),
          axis.text.y = element_text(size=14),
          legend.position="bottom", 
          legend.title = element_blank(),
          text = element_text(size=14), 
          legend.text=element_text(size=14), 
          axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))
  }


## ----------------------------------------------------------##


overlaid_rates_plot_nbeac <- function(df){ 
  ggplot(df) +
    labs(x = "\nPosition (cm)", y = "Firing rate (Hz)", color = "Cluster id") +
    geom_smooth(aes(x=Position, y=asr_b_smooth, group=new_cluster_id), size=0.3, weight=0.5, alpha=0.5, colour="grey71", span = 0.2, method = 'loess', se=FALSE) +
    geom_smooth(aes(x=Position, y=asr_nb_smooth, group=new_cluster_id), size=0.3, weight=0.5, alpha=0.5, colour="grey71", span = 0.2, method = 'loess', se=FALSE) +
    geom_smooth(aes(x=Position, y=asr_nb_smooth), size=0.8, weight=1, alpha=0.5, colour="blue2", span = 0.2, method = 'loess', se=TRUE) +
    geom_smooth(aes(x=Position, y=asr_b_smooth), size=0.8, weight=1, alpha=0.5, colour="red2", span = 0.2, method = 'loess', se=TRUE) +
    guides(fill=FALSE) +
    theme_classic() +
    theme(axis.text.x = element_text(size=14),
          axis.text.y = element_text(size=14),
          legend.position="bottom", 
          legend.title = element_blank(),
          text = element_text(size=14), 
          legend.text=element_text(size=14), 
          axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))
}



## ----------------------------------------------------------##

change_in_firing_rate <- function(data, startbin = 30, endbin = 170) {
  rates_tibble <- as_tibble(data)
  bins=c(startbin:endbin)
  change <- max(rates_tibble$Rates) - min(rates_tibble$Rates)
  }


change_in_location <- function(data, startbin = 30, endbin = 170) {
  rates_tibble <- as_tibble(data)
  locs <- which.max(rates_tibble$Rates) - which.min(rates_tibble$Rates)
  }





plot_comparisons <- function(data, column1, column2){
  ggplot(data, aes(x = column1, y = column2)) + 
  geom_point() +
  theme_minimal()
  }



plot_comparisons_w_color <- function(data, column1, column2, column3){
  ggplot(data, aes(x = column1, y = column2, color=column3)) + 
    geom_point() +
    theme_classic() +
    theme(axis.text.x = element_text(size=15),
          axis.text.y = element_text(size=15),
          legend.position="bottom", 
          legend.title = element_blank(),
          text = element_text(size=15), 
          legend.text=element_text(size=15), 
          axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))) 
  }









### These functions find the average firing rate in the start and end of track black boxes

end_black_box_helper <- function(df) {
  data = as.tibble(df)
  max_bb <- mean(data$Rates[0:30], na.rm = FALSE)
  }

start_black_box_helper <- function(df) {
  data = as.tibble(df)
  max_bb <- mean(data$Rates[170:195], na.rm = FALSE)
  if(max_bb > 100)
    max_bb <- (as.numeric(50))
    
  }




## Gives count, mean, standard deviation, standard error of the mean, and confidence interval (default 95%) of selected column in dataframe by selected groups.
##   data: a data frame.
##   measurevar: the name of a column that contains the variable to be summariezed
##   groupvars: a vector containing names of columns that contain grouping variables
##   na.rm: a boolean that indicates whether to ignore NA's
##   conf.interval: the percent range of the confidence interval (default is 95%)
summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
  #library(plyr)
  
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  
  # This does the summary. For each group's data frame, return a vector with
  # N, mean, and sd
  datac <- ddply(as.numeric(unlist(data)), groupvars, .drop=.drop,
                 .fun = function(xx, col) {
                   c(N    = length2(xx[[col]], na.rm=na.rm),
                     mean = mean   (xx[[col]], na.rm=na.rm),
                     sd   = sd     (xx[[col]], na.rm=na.rm)
                   )
                 },
                 measurevar
  )
  
  # Rename the "mean" column    
  datac <- rename(datac, c("mean" = measurevar))
  
  datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval: 
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + .5, datac$N-1)
  datac$ci <- datac$se * ciMult
  
  return(datac)
}



mean_function <- function(all_ramp_df, meanvar=meanvar){ 
  gd <- all_ramp_df %>% 
    group_by(ramp_id) %>% 
    summarise(meanvar = mean(meanvar))
}

