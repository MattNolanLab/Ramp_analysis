### This script contains functions which are called in the Analysis markdown code


## Adds positions to a single column data frame that contains a neurons binned mean firing rate
add_position <- function(df, session_id = "", cluster_id = "") {
  len = length(unlist(df))
  df <- tibble(Rates = unlist(df), Position = rep(1:len)) 
  #if(all(is.na(df$Rates))){print(paste0("All NAs. Session: ", session_id, ". Cluster:", cluster_id))}
  df
}



## Fits a linear model to firing rate data contained in df, across the range from startbin to endbin.
# Returns output as a dataframe containing key model parameters extracted with glance and tidy.
lm_tidy_helper <- function(df,
                           startbin,
                           endbin) {
  # Check for NAs
  if (all(is.na(df$Rates))) {
    df <-
      tibble(
        r.squared = c(NA),
        p.value = c(NA),
        intercept = c(NA),
        slope = c(NA)
      )
    return(df)
  }
  
  df <- df %>%
    subset(Position >= startbin & Position <= endbin)
  df_fit <- lm(Rates ~ Position, data = df, na.action = na.exclude)
  
  # get the model parameters
  params <- select(glance(df_fit), r.squared, p.value)
  # get the coefficients
  coeffs <- tidy(df_fit)
  # combine the parameters and coefficients
  params$intercept <- coeffs$estimate[[1]]
  params$slope <- coeffs$estimate[[2]]
  return(params)
}

###Function to generate shuffled datasets from a neuron's mean firing rate profile.
# shuffles defines the number of shuffles. The default value is for testing.
# Use a larger value for analyses, e.g. 1000.
# shuffles spikes using sample() function
# fits lm
# extracts coefficients
# stores coefficients for each 1000 shuffles (less memory than saving 1000 shuffles)
# The id, slope, r2 and p values for each shuffled dataset are returned.

shuffle_rates <- function(df, startbin, endbin, shuffles = 10) {
  df_modified <- data.frame(neuron=as.numeric(),
                            slope=as.numeric(), 
                            rsquared=as.numeric(), 
                            pval=vector())
  names(df_modified) <- c("neuron", "slope", "rsquared", "pval")
  x <- 0
  repeat {
    shuff_df <- tibble(Rates = sample(as.vector(unlist(df)),replace = TRUE, prob = NULL), Position = c(1:200))
    df_mod <- lm_tidy_helper(shuff_df, startbin, endbin)
    data <- data.frame(as.numeric(x), df_mod$slope, df_mod$r.squared, df_mod$p.value)
    names(data) <- c("neuron", "slope", "r.squared", "p.value")
    df_modified <- rbind(df_modified,data)
    
    x = x+1
    if (x == shuffles){ 
      break
    }
  }
  return(df_modified)
}




# Functions to return the slope for a given quantile in the shuffled datasets 
extract_quantile_shuffle_slopes <- function(df, q_prob = 0.05){
  df <- tibble(slopes = unlist(df$slope), r.squared = unlist(df$r.squared))
  if (all(is.na(df$slopes))) {
    return(NA)
  }
  min_slope_o <- quantile(as.numeric(unlist(df$slopes)), c(q_prob))[1]
  return(min_slope_o)
}



# To classify neurons based on:
# 1. Whether their slopes are outside the 5-95% range of the shufled data.
# 2. Whether the adjusted p-value of the linear model fit is <= 0.01.
compare_slopes <-
  function(min_slope = 1,
           max_slope = 1,
           slope = 1,
           pval = 1) {
    if (any(is.na(list(min_slope, max_slope, slope, pval)))) {
      return("Unclassified")
    }
    if (pval > 0.01) {
      return("Unclassified")
    } else if (slope < min_slope & pval < 0.01) {
      return("Negative")
    } else if (slope > max_slope & pval < 0.01) {
      return("Positive")
    } else {
      return("Unclassified")
    }
  }

#Calculates the difference between mean rate and predicted mean rate at the start of the homebound zone
calc_predict_diff <- function(rates, fit)
{
  diff <- mean(as.double(rates[110:115])) - mean(as.double(fit[110:115]))
}

#make function to predict firing rate
lm_predict <- function(df){
  new.data <- data.frame(Position =df$Position)
}

# Predict mean and confidence intervals for firing rate at the start of the homebound zone (track positions 110 to 115 cm) based on firing in the outbound zone (30 to 90 cm).
predict_homebound <- function(df, fit_start = 30, fit_end = 90, predict_start = 110, predict_end = 115){
  # check for NAs
  if(all(is.na(df))) 
    return(NA)
  # Make track column
  df <- tibble(Rates = unlist(df), Position=rep(1:200))
  # fit
  model <- lm(Rates ~ Position, data = filter(df, Position >= fit_start, Position <= fit_end))
  # predict
  homebound_prediction_pos <- tibble(Position = rep(1:200))
  homebound_prediction <- predict(model, newdata = homebound_prediction_pos, interval = "prediction", level = 0.99) 
  as.tibble(homebound_prediction)
}


#Test whether data lies outside of confidence intervals
offset_test <- function(rates, lwr, upr){
  # check for NAs
  if(all(is.na(rates))) 
    return(NA)
  rates <- mean(as.double(rates[110:115]))
  upr <- mean(as.double(upr[110:115]))
  lwr <- mean(as.double(lwr[110:115]))
  if(rates > upr) {
    return("Pos")
  }
  
  if (rates <= lwr) {
    return("Neg")
  }
  return("None")
  
}

#write function to mark cells based on groups
mark_track_category <- function(outbound, homebound){
  if( outbound == "Positive" & homebound == "Negative") {
    return( "posneg" ) 
  } else if( outbound == "Positive" & homebound == "Positive") {
    return( "pospos" )
  } else if( outbound == "Negative" & homebound == "Positive") {
    return( "negpos" )
  } else if( outbound == "Negative" & homebound == "Negative") {
    return( "negneg" )
  } else if( outbound == "Negative" & homebound == "Unclassified") {
    return( "negnon" )
  } else if( outbound == "Positive" & homebound == "Unclassified") {
    return( "posnon" )
  } else {
    return("None")
  }
}

# write function to mark cells based on groups
mark_numeric_track_category <- function(outbound, homebound){
  if( outbound == "Positive" & homebound == "Negative") {
    return( as.numeric(2) ) 
  } else if( outbound == "Positive" & homebound == "Positive") {
    return( as.numeric(1) )
  } else if( outbound == "Negative" & homebound == "Positive") {
    return( as.numeric(5) )
  } else if( outbound == "Negative" & homebound == "Negative") {
    return( as.numeric(4) )
  } else if( outbound == "Negative" & homebound == "Unclassified") {
    return( as.numeric(6) )
  } else if( outbound == "Positive" & homebound == "Unclassified") {
    return( as.numeric(3) )
  } else {
    return(as.numeric(0))
  }
}

#Function to classify neurons based on offset 
mark_reset_group_predict <- function(offset){
  if (is.na(offset) ) {
    return( "None" )
  } else if( offset == "None") {
    return( "Continuous" )
  } else if( ( offset == "Neg" ||  offset == "Pos")) {
    return( "Reset" ) 
  }
}

# Load circular shuffled data from Python.
# df is the master data frame, e.g. spatial_firing.
# The function returns a version of df with the resilts appended.
load_circ_shuffles <- function(df, cs_path) {
  shuffled_df <- read_feather(cs_path)
  
  # I will put the results from Python here (copy of previous results over the existing ones)
  df$spike_shuffle_results_b_o <- df$shuffle_results_b_o  # I did this because this 'nested' format is needed
  df$spike_shuffle_results_nb_o <- df$shuffle_results_nb_o  # I did this because this 'nested' format is needed
  df$spike_shuffle_results_p_o <- df$shuffle_results_p_o  # I did this because this 'nested' format is needed
  df$row_names <- seq(1, nrow(df), by=1)
  
  # get list of cells based on session id + cluster id
  # add unique id for each cell to both data frames
  shuffled_df$unique_cell_id <- paste(shuffled_df$session_id, shuffled_df$cluster_id)
  df$unique_cell_id <- paste(df$session_id, df$cluster_id)
  unique_cells = unique(shuffled_df[c("unique_cell_id")])
  number_of_cells = nrow(unique_cells)
  print('Number of cells in spike-level shuffle data:')
  print(number_of_cells)
  
  # iterate on list of cells and change 'spike_shuffle_results_b_o' column
  for(i in 1:nrow(unique_cells)) {
    # these are the rows that correspond to the cell
    cell_rows <- shuffled_df %>% filter(unique_cell_id == toString(unique_cells[i,]))
    # get part of df that corresponds to shuffled data from the cell from outbound
    shuffled_results_b <- cell_rows %>% select(beaconed_r2_ob, beaconed_slope_ob, beaconed_p_val_ob)
    shuffled_results_nb <- cell_rows %>% select(non_beaconed_r2_ob, non_beaconed_slope_ob, non_beaconed_p_val_ob)
    shuffled_results_p <- cell_rows %>% select(probe_r2_ob, probe_slope_ob, probe_p_val_ob)
    
    # rename the collumns in shuffled_results to "slope", "r.squared", "p.value"
    names(shuffled_results_b)[names(shuffled_results_b) == "beaconed_r2_ob"] <- "r.squared"
    names(shuffled_results_b)[names(shuffled_results_b) == "beaconed_slope_ob"] <- "slope"
    names(shuffled_results_b)[names(shuffled_results_b) == "beaconed_p_val_ob"] <- "p.value"
    names(shuffled_results_nb)[names(shuffled_results_nb) == "non_beaconed_r2_ob"] <- "r.squared"
    names(shuffled_results_nb)[names(shuffled_results_nb) == "non_beaconed_slope_ob"] <- "slope"
    names(shuffled_results_nb)[names(shuffled_results_nb) == "non_beaconed_p_val_ob"] <- "p.value"
    names(shuffled_results_p)[names(shuffled_results_p) == "probe_r2_ob"] <- "r.squared"
    names(shuffled_results_p)[names(shuffled_results_p) == "probe_slope_ob"] <- "slope"
    names(shuffled_results_p)[names(shuffled_results_p) == "probe_p_val_ob"] <- "p.value"
    
    neuron_list <- seq(1, nrow(shuffled_results_b), by=1) # make list [1...1000]
    shuffled_results_b$neuron=neuron_list  # add 'neuron' column with shuffle ids
    shuffled_results_nb$neuron=neuron_list  # add 'neuron' column with shuffle ids
    shuffled_results_p$neuron=neuron_list  # add 'neuron' column with shuffle ids
    
    # find cell in ramp data frame that we are updating the shuffled for
    cell_index <- df[df$unique_cell_id==toString(unique_cells[i,]),]$row_names
    # put spike shuffle results in R df
    df$spike_shuffle_results_b_o[cell_index] <- list(shuffled_results_b)
    df$spike_shuffle_results_nb_o[cell_index] <- list(shuffled_results_nb)
    df$spike_shuffle_results_p_o[cell_index] <- list(shuffled_results_p)
  }
  
  # Now I renamed the spike_shuffle_results_b_o, spike_shuffle_results_nb_o, spike_shuffle_results_p_o to
  #                         shuffle_results_b_o,       shuffle_results_nb_o,       shuffle_results_p_o
  # first drop the old columns
  df <- df[,!grepl("^shuffle_results_b_o",names(df))]
  df <- df[,!grepl("^shuffle_results_nb_o",names(df))]
  df <- df[,!grepl("^shuffle_results_p_o",names(df))]
  
  # then rename the newly created columns
  names(df)[names(df) == "spike_shuffle_results_b_o"] <- "shuffle_results_b_o"
  names(df)[names(df) == "spike_shuffle_results_nb_o"] <- "shuffle_results_nb_o"
  names(df)[names(df) == "spike_shuffle_results_p_o"] <- "shuffle_results_p_o"
  df
col}


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

