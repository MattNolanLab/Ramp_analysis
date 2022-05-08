### This script contains functions which are called in the Analysis markdown code

### Functions here are initially used in Figure 1 code

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


# Function to normalize firing rates
normalise_rates <- function(df){
  df <- tibble(Rates = unlist(df), Position = rep(1:200))
  x <- scale(df$Rates, center=TRUE, scale=TRUE)[,1]
  return(x)
}

normalise_smooth_rates <- function(df){
  df <- tibble(Rates_smoothed = unlist(df), Position = rep(1:200))
  x <- scale(df$Rates_smoothed, center=TRUE, scale=TRUE)[,1]
  return(x)
}

#C alculates the difference between mean rate and predicted mean rate at the start of the homebound zone
calc_predict_diff <- function(rates, fit)
{
  diff <- mean(as.double(rates[110:115])) - mean(as.double(fit[110:115]))
}

#make function to predict firing rate
lm_predict <- function(df){
  new.data <- data.frame(Position =df$Position)
}

# Predict mean and confidence intervals for firing rate at the start of the homebound zone (track positions 110 to 115 cm) based on firing in the outbound zone (30 to 90 cm).
predict_homebound <- function(df, fit_start = 30, fit_end = 90){
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
  as_tibble(homebound_prediction)
}


#Test whether data lies outside of confidence intervals
offset_test <- function(rates, lwr, upr, predict_start = 110, predict_end = 115){
  # check for NAs
  if(all(is.na(rates))) 
    return(NA)
  rates <- mean(as.double(rates[predict_start:predict_end]))
  upr <- mean(as.double(upr[predict_start:predict_end]))
  lwr <- mean(as.double(lwr[predict_start:predict_end]))
  if(rates > upr) {
    return("Pos")
  }
  
  if (rates <= lwr) {
    return("Neg")
  }
  return("None")
  
}

# Function to give a text label to cells based on groups
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

# Function to give a numeric label to cells based on groups
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



# Plot histogram of distribution of firing rate offsets
offset_ggplot <- function(df, diff_colname = "predict_diff", group_colname = "reset_group", colour_1 = "grey", colour_2 = "chartreuse3", colour_3 = "red") {
  ggplot(data=df, aes(x = unlist(.data[[diff_colname]]), fill=as.factor(unlist(.data[[group_colname]])))) +
    coord_cartesian(xlim=c(-6,6)) +
    geom_histogram(aes(y=..count..), alpha=0.5) +
    scale_fill_manual(values=c(colour_1, colour_2, colour_3)) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) +
    labs(y="Density", x="") +
    theme_classic() +
    theme(axis.text.x = element_text(size=13),
          axis.text.y = element_text(size=13),
          legend.position="bottom", 
          legend.title = element_blank(),
          text = element_text(size=13), 
          legend.text=element_text(size=13), 
          axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))
}


# Plot mean and SEM of firing rate as a function of position.
add_track <- function(gg, xlab = "Location (cm)", ylab = "Stops (cm)") {
  gg +
    annotate("rect", xmin=-30, xmax=0, ymin=-Inf,ymax=Inf, alpha=0.2, fill="Grey60") +
    annotate("rect", xmin=140, xmax=170, ymin=-Inf,ymax=Inf, alpha=0.2, fill="Grey60") +
    annotate("rect", xmin=60, xmax=80, ymin=-Inf,ymax=Inf, alpha=0.2, fill="Chartreuse4") +
    scale_x_continuous(breaks=seq(-30,170,100), expand = c(0, 0)) +
    labs(y = ylab, x = xlab) +
    theme_classic() +
    theme(axis.text.x = element_text(size=18),
          axis.text.y = element_text(size=18),
          legend.title = element_blank(),
          text = element_text(size=18),
          plot.margin = margin(21, 25, 5, 20))
}


mean_SEM_plots_prep <- function(df) {
  df <- df %>% dplyr::summarise(mean_r = mean(Rates, na.rm = TRUE),
                                sem_r = std.error(Rates, na.rm = TRUE))
}

mean_SEM_plots <- function(df, colour1 = "blue"){
  gg <- ggplot(data=df) +
    geom_ribbon(aes(x=Position, y=mean_r, ymin = mean_r - sem_r, ymax = mean_r + sem_r), fill = colour1, alpha=0.2) +
    geom_line(aes(y=mean_r, x=Position), color = colour1)
  add_track(gg, xlab = "Position", ylab = "Z-scored firing rate")
}

## --------------------------------------------------------------------------------------------- ##
# Load circular shuffled data from Python.

# this function will take load shuffles for a single trial type, e.g. beaconed outbound
# The input dataframe is the 
local_circ_shuffles <- function(df_in, cs_path) {
  shuffled_df <- read_feather(cs_path)
  
  # get list of cells based on session id + cluster id
  # add unique id for each cell to both data frames
  shuffled_df$unique_cell_id <- paste(shuffled_df$session_id, shuffled_df$cluster_id)
  unique_cells = unique(shuffled_df[c("unique_cell_id")])
  number_of_cells = nrow(unique_cells)
  print('Number of cells in spike-level shuffle data:')
  print(number_of_cells)
  
  # Provides a reference for cell IDs in the experimental data
  unique_cell_ids <- paste(df_in$session_id, df_in$cluster_id)
  
  shuffled_df <- shuffled_df %>%
    group_by(unique_cell_id) %>%
    nest()
  
  # select shuffled data that matches the experimental data
  shuffled_df_select <- shuffled_df[shuffled_df$unique_cell_id %in% unique_cell_ids, ]

  shuffled_df_select <- unnest(shuffled_df_select, cols = c(data))
  
  # reformat shuffled data
  shuffled_b <- shuffled_df_select %>%
    select(unique_cell_id, shuffle_id, beaconed_r2_ob, beaconed_slope_ob, beaconed_p_val_ob) %>%
    rename(neuron = "shuffle_id", slope = "beaconed_slope_ob", r.squared = "beaconed_r2_ob", p.value = "beaconed_p_val_ob") %>%
    group_by(unique_cell_id) %>%
    nest()
  shuffled_nb <- shuffled_df_select %>%
    select(unique_cell_id, shuffle_id, non_beaconed_r2_ob, non_beaconed_slope_ob, non_beaconed_p_val_ob) %>%
    rename(neuron = "shuffle_id", slope = "non_beaconed_slope_ob", r.squared = "non_beaconed_r2_ob", p.value = "non_beaconed_p_val_ob") %>%
    group_by(unique_cell_id) %>%
    nest()
  shuffled_p <- shuffled_df_select %>%
    select(unique_cell_id, shuffle_id, probe_r2_ob, probe_slope_ob, probe_p_val_ob) %>%
    rename(neuron = "shuffle_id", slope = "probe_slope_ob", r.squared = "probe_r2_ob", p.value = "probe_p_val_ob") %>%
    group_by(unique_cell_id) %>%
    nest()
  shuffled_b_h <- shuffled_df_select %>%
    select(unique_cell_id, shuffle_id, beaconed_r2_hb, beaconed_slope_hb, beaconed_p_val_hb) %>%
    rename(neuron = "shuffle_id", slope = "beaconed_slope_hb", r.squared = "beaconed_r2_hb", p.value = "beaconed_p_val_hb") %>%
    group_by(unique_cell_id) %>%
    nest()
  shuffled_nb_h <- shuffled_df_select %>%
    select(unique_cell_id, shuffle_id, non_beaconed_r2_hb, non_beaconed_slope_hb, non_beaconed_p_val_hb) %>%
    rename(neuron = "shuffle_id", slope = "non_beaconed_slope_hb", r.squared = "non_beaconed_r2_hb", p.value = "non_beaconed_p_val_hb") %>%
    group_by(unique_cell_id) %>%
    nest()
  shuffled_p_h <- shuffled_df_select %>%
    select(unique_cell_id, shuffle_id, probe_r2_hb, probe_slope_hb, probe_p_val_hb) %>%
    rename(neuron = "shuffle_id", slope = "probe_slope_hb", r.squared = "probe_r2_hb", p.value = "probe_p_val_hb") %>%
    group_by(unique_cell_id) %>%
    nest()
  
  
  df <- tibble(unique_cell_id = shuffled_b$unique_cell_id,
               shuffle_results_b_o = shuffled_b$data,
               shuffled_results_nb_o = shuffled_nb$data,
               shuffled_results_p_o = shuffled_p$data,
               shuffle_results_b_h = shuffled_b_h$data,
               shuffled_results_nb_h = shuffled_nb_h$data,
               shuffled_results_p_h = shuffled_p_h$data)
  
  return(df)
}


## ----------------------------------------------------------##

### Functions below here are initially used in Figure 2 code.

# Function to fit general linear mixed effect models
# The goal here is to evaluate influences of position, speed and acceleration on firing rate.
# This is set up with family = poisson as the data binned in time are counts.
# For use with smoothed data would need to change this to gamma (which won't work where there are zeros)
# Or consider using Tweedie family implemented in the glmBB package.
# Note also that we get similar results using linear mixed effect models so GLMER while more 'correct' may not be necessary.
# TT is trial type. Codes used in the spatial_firing$spikes_in_time are 0 (beaconed), 1 (non-beaconed) and 2 (probe).
# thresh is the minimum number of trials required for fitting the data.
mm_fit <- function(df, TT = 0, thresh = 20) {
  if (length(df) == 1){
    return(NA)}
  df <-
    tibble(
      Rates = as.numeric(df[[1]]),
      Position = as.numeric(df[[2]]),
      Acceleration = as.numeric(df[[4]]),
      Speed = as.numeric(df[[3]]),
      Trials = as.factor(df[[5]]), 
      Types = as.factor(df[[6]])
    )
  print(nrow(df))
  
  # subset by position, speed, types (beaconed/probe) and remove rates < 0.01 as model does not like this
  df <- df %>%
    subset(Position >= 30 & Position <= 90 & Speed >= 3 & Types == TT) %>%
    select(-Types) 
  
  #scale variables, do not center or values go below 0 which does not work for this gamma model
  df$Acceleration <- scale(df$Acceleration, center=FALSE, scale=TRUE)
  df$Rates <- scale(df$Rates, center=FALSE, scale=TRUE)
  df$Speed <- scale(df$Speed, center=FALSE, scale=TRUE)
  df$Position <- scale(df$Position, center=FALSE, scale=TRUE)
  

  
  if (length(df) == 1 | nrow(df) < thresh) {
    return(NA)
  }
  
  # return NA if variables contain any NAs after scaling (very possible if cell doesn't spike on rewarded trials)
  if (sum(is.na(as.matrix(df))) > 0) {
    return(NA)
  }
  
  df_int <- lme4::glmer(formula = Rates ~ Position + Speed + Acceleration + (1 + Position | Trials), 
                        data = df,
                        na.action = na.exclude,
                        family = poisson(link = "log"),
                        control=lme4::glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
  }

# Function to extract P values for each coefficient from the model
mm_function <- function(mm, session_id) {
  if (is.na(mm)) {
    return(tibble(pos = NA, speed = NA, accel = NA))
  }
  
  modelAnova <- car::Anova(mm)
  return_tibble <- tibble(pos = modelAnova$"Pr(>Chisq)"[[1]],
                          speed = modelAnova$"Pr(>Chisq)"[[2]],
                          accel = modelAnova$"Pr(>Chisq)"[[3]])
}


# Helper function for extracting P values for each coefficient from the model
mm_pvalues <- function(mm, session_id) {
  tryCatch({
    mm_function(mm, session_id)
  },
  error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}



# Helper function to link to general linear mixed model fit
mm_fit_function <- function(mm, TT = 0, thresh = 20) {
  tryCatch({
    mm_fit(mm,TT, thresh)
  },
  error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}


## Categorize neurons based on significant model coefficients
coef_comparison <- function(null_pos, null_speed, null_accel, pval = 0.01){
  if(is.na(null_pos)) {
    return("None")
    
  } else if( null_pos < pval & null_accel > pval & null_speed > pval) {
    return( "P" )
    
  } else if( null_pos > pval & null_accel > pval & null_speed < pval) {
    return( "S" ) 
    
  } else if( null_pos > pval & null_accel < pval & null_speed > pval) {
    return( "A" )
    
  } else if( null_pos < pval & null_accel > pval & null_speed < pval) {
    return("PS")
    
  } else if( null_pos < pval & null_accel < pval & null_speed > pval) {
    return( "PA" )
    
  } else if( null_pos > pval & null_accel < pval & null_speed < pval) {
    return("SA")
    
  } else if( null_pos < pval & null_accel < pval & null_speed < pval) {
    return("PSA")
    
  } else {
    return("None")
  }
}


# Function to calculate standardized coefficients for a LMER
#https://stackoverflow.com/questions/25142901/standardized-coefficients-for-lmer-model 
stdCoef.merMod <- function(object) {
  sdy <- sd(getME(object,"y"))
  sdx <- apply(getME(object,"X"), 2, sd)
  sc <- fixef(object)*sdx/sdy
  se.fixef <- coef(summary(object))[,"Std. Error"]
  se <- se.fixef*sdx/sdy
  return(data.frame(stdcoef=sc, stdse=se))
}

# Helperfunction to calculate standardized coefficients from the model fits
std_coef <- function(mm) {
  tryCatch({
  mod <- stdCoef.merMod(mm)
  mod_coefs <- tibble(pos = mod[2,1],
                      speed = mod[3,1],
                      accel = mod[4,1])
      },
    error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}

# Extracts counts for numbers of neurons sorted according to the classification from the GLMER fit
make_coeffs_table <- function(df) {
  df <- df %>%
    unlist() %>%
    table() %>%
    as_tibble() %>%
    mutate(perc = (n / sum(n)) * 100)
  colnames(df) <- c("ramp_id", "num", "perc")
  df
}


# Function for plotting standardised coefficients
# df should contain columns with coef_type and coef
# These columns are generated as outputs after GLMER fits
# For Figure 2 these columns are in spatial_firing_coefs.
# This data frame was generated by unnesting spatial_firing.
standard_plot <- function(df) {
  level_order <- c("P", "S", "A")
  ggplot(data=df, aes(x = factor(coef_type), y = as.numeric(coef))) +
    geom_violin(aes(x = factor(coef_type), y = as.numeric(coef), fill=factor(coef_type, level=level_order)), alpha=0.7) +
    stat_summary(fun=mean, geom="point", shape=23, size=2) +
    geom_jitter(alpha=0.05) +
    geom_hline(yintercept=0, linetype="dashed", color = "black") +
    scale_fill_manual(values=c("firebrick1","gold","dodgerblue2")) +
    labs(y = "std coef", x="\n model parameter") +
    scale_y_continuous(trans=pseudolog10_trans) +
    theme_classic() +
    theme(axis.text.x = element_text(size=14),
          axis.text.y = element_text(size=12),
          legend.position="bottom", 
          legend.title = element_blank(),
          text = element_text(size=12), 
          legend.text=element_text(size=12), 
          axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))) 
}



## ----------------------------------------------------------##

### Functions below here are initially used in Figure 3 code.


# A function that extracts into a tibble data from spatial_firing$spikes_in_time_reward_hit/run/try
extract_to_tibble <- function(df) {
  df <-
    tibble(
      Rates = as.numeric(Re(df[[1]])),
      Position = as.numeric(Re(df[[3]])),
      Trials = as.numeric(Re(df[[4]]))
    )
  return (df)
}



# Function to join firing rates from different trial types and add indicator of the type of trial
# The trial types are:
# spatial_firing$spikes_in_time_reward_hit
# spikes_in_time_reward_run
# spikes_in_time_reward_try
join_rates <- function(hit, run, try, session_id, cluster_id) {
  if (any(is.na(hit)) | any(is.na(run)) | any(is.na(try))) { 
    return(NA)
  }
  
  hit_df <- extract_to_tibble(hit)
  run_df <- extract_to_tibble(run)
  try_df <- extract_to_tibble(try)
  
  df <- tibble(Rates = c(hit_df$Rates, 
                         run_df$Rates,
                         try_df$Rates), 
               Reward_indicator = c(rep("Rewarded", times=nrow(hit_df)), 
                                    rep("Run", times=nrow(run_df)), 
                                    rep("Try", times=nrow(try_df))), 
               Position = c(hit_df$Position, 
                            run_df$Position, 
                            try_df$Position), 
               Trials = c(hit_df$Trials,
                          run_df$Trials,
                          try_df$Trials))
  
  return (df)
}


# Function to fit a linear mixed effect model that takes trial type (hit, try, run) into account
# The car package is used to extract slope significance
compare_models_slope_lm <- function(df, run, try){
  tryCatch({
    if (any(is.na(run)) | any(is.na(try)) | any(is.na(df))) { 
      return("Contains NAs")
    }
    if (length(df) == 1 | nrow(df) < 6)
      return("Too small")
    
    df <- df %>%
      filter(Position >= 30, Position <= 90) %>%
      mutate(Rates = scale(Rates)) # This doesn't seem to make any substantial difference
    fit <- lme4::lmer(Rates ~ scale(Position) * Reward_indicator + (1+scale(Position) | Trials), data = df, na.action=na.omit)
    modelAnova <- car::Anova(fit)
    to_return <- modelAnova$"Pr(>Chisq)"[[3]]
  },
  error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}

# Function to fit a general linear mixed effect model that takes trial type (hit, try, run) into account
# This is set up with family = poisson as the data binned in time are counts.
# For use with smoothed data would need to change this to gamma (which won't work where there are zeros)
# Or consider using Tweedie family implemented in the glmBB package.
# Note also that we get similar results using linear mixed effect models so GLMER while more 'correct' may not be necessary.
# The car package is used to extract slope significance
compare_models_slope_glm <- function(df, run,try){
  tryCatch({
    if (any(is.na(run)) | any(is.na(try)) | any(is.na(df))) { 
      return("Contains NAs")
    }
    if (length(df) == 1 | nrow(df) < 6)
      return("Too small")
    
    df <- df %>%
      filter(Position >= 30, Position <= 90)
    
    glm1 <- glm(Rates ~ Position * Reward_Indicator , family = poisson(link = "log"), data = df)
    
    fit <- lme4::glmer(formula = Rates ~ Position * Reward_indicator + (1 + Position | Trials), 
                       data = df, 
                       na.action = na.exclude,
                       family = poisson(link = "log"),
                       start=list(fixef=coef(glm1)),
                       control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
    
    modelAnova <- car::Anova(fit)
    to_return <- modelAnova$"Pr(>Chisq)"[[3]]
  },
  error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}


# Function to classify neurons based on whether there is a significant effect of trial outcome
mark_neurons_sig <- function(pval){
  tryCatch({
    if (is.na(pval)) {
      return(NA)
    }
    if (pval == "NULL") {
      return(NA)
    }
    if( pval < 0.01) {
      return( "Significant" )
    } else if( pval >= 0.01) {
      return( "Not-Significant" )
    } else {
      return("None")
    }
  },
  error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
  
}


# Function to join average firing rates from different trial types, normalize the rates and add indicator of the type of trial
join_average_rates <- function(hit, run, try, session_id, cluster_id) {
  if (any(is.na(hit)) | any(is.na(run)) | any(is.na(try))) { 
    return(
      df <- tibble(Rates = rep(NA, times=600), 
                   Position = rep(NA, times=600),
                   Reward_indicator = c(rep("Rewarded", times=200), rep("Run Through", times=200), rep("Try", times=200)))
    )
  }
  df <- tibble(Rates = c(unlist(hit), unlist(run), unlist(try)),
               Position = c(rep(-30:169), rep(-30:169), rep(-30:169)),
               Reward_indicator = c(rep("Rewarded", times=200), rep("Run Through", times=200), rep("Try", times=200)))
  df$Rates <- scale(df$Rates, center=TRUE, scale=TRUE)[,1]
  return (df)
}


# Generic function to subset data ready for plotting with mean_SEM_plots_by_Outcome
# Note the unnest() is only carried out if there is data.
subset_for_plots <- function(df, outbound_class = "Positive", homebound_class = "Positive") {
  df <- df %>%
    filter(lm_group_b == outbound_class,
           lm_group_b_h == homebound_class) %>%
    select(avg_both_asr_b) %>%
    when(nrow(.) != 0 ~ unnest(., c(avg_both_asr_b))
    )
}

# Generic function to plot firing rate ± SEM as a function of position and colour coded according to trial outcome.
# The function expects to receive the unnested mean firing rates for all neurons that are to be plotted.
# Conditions for selection should be given before calling the function.
# Column names of the input data frame should be 'Rates', 'Position' and 'Reward_indicator'.
mean_SEM_plots_by_Outcome <- function(df, x_start = -30, x_end = 170) {
  # Check for data
  if(is.null(df) == TRUE) {return(
    ggplot() + theme_void())
    } 
  # check for all NAs
  if (sum(is.na(df$Rates)) == dim(df)[[1]]) {
    return(ggplot() + theme_void())
  }
  # Carry on
  df <- df %>%
    group_by(Position, Reward_indicator) %>%
    dplyr::summarise(mean_b = mean(Rates, na.rm = TRUE),
                     se_b = std.error(Rates, na.rm = TRUE))
  
  ggplot(data=df) +
    annotate("rect", xmin=-30, xmax=0, ymin=-2,ymax=Inf, alpha=0.2, fill="Grey60") +
    annotate("rect", xmin=140, xmax=170, ymin=-2,ymax=Inf, alpha=0.2, fill="Grey60") +
    annotate("rect", xmin=60, xmax=80, ymin=-2,ymax=Inf, alpha=0.2, fill="Chartreuse4") +
    geom_ribbon(aes(x=Position, y=mean_b, ymin = mean_b - se_b, ymax = mean_b + se_b,
                    fill=factor(Reward_indicator)), alpha=0.1) +
    geom_line(aes(y=mean_b, x=Position, color=factor(Reward_indicator)), alpha=0.5) +
    scale_fill_manual(values=c("black", "red", "blue")) +
    scale_color_manual(values=c("black", "red", "blue")) +
    #labs(y = "Mean firing rate (Hz)", x = "Location (cm)") +
    labs(y = "Z-scored firing rate", x = "Location (cm)") +
    #xlim(-30, 170) +
    theme_classic() +
    scale_x_continuous(breaks=seq(-30,170,100), expand = c(0, 0)) +
    theme(axis.text.x = element_text(size=14),
          axis.text.y = element_text(size=14),
          legend.title = element_blank(),
          legend.position="none", 
          text = element_text(size=14),
          plot.margin = margin(21, 25, 5, 20))
}

# Function to generate plots by trial outome for each class of ramping neurons
# Also returns the number of neurons that contribute to each plot
# Calculation of number of neurons without NAs is somewhat improvised
# This could be modified by including neuron ID, etc.
all_plots_by_outome <- function(df) {
  
  NegNegNeurons <- df %>%
    subset_for_plots("Negative", "Negative")
  NegNeg_plot <- NegNegNeurons %>%
    mean_SEM_plots_by_Outcome(-29,169)
  NegNeg_N <- sum(!is.na(NegNegNeurons$Rates))/600 # Relies on their being 200 x 3 location points
  
  NegPosNeurons <- df %>%
    subset_for_plots("Negative", "Positive")
  NegPos_plot <- NegPosNeurons %>%
    mean_SEM_plots_by_Outcome(-29,169)
  NegPos_N <- sum(!is.na(NegPosNeurons$Rates))/600 
  
  PosPosNeurons <- df %>%
    subset_for_plots("Positive", "Positive")
  PosPos_plot <- PosPosNeurons %>%
    mean_SEM_plots_by_Outcome(-29,169)
  PosPos_N <- sum(!is.na(PosPosNeurons$Rates))/600 
  
  PosNegNeurons <- df %>%
    subset_for_plots("Positive", "Negative")
  PosNeg_plot <- PosNegNeurons %>%
    mean_SEM_plots_by_Outcome(-29,169)
  PosNeg_N <- sum(!is.na(PosNegNeurons$Rates))/600 
  
  
  return(list(list(NegNeg_plot, NegPos_plot, PosPos_plot, PosNeg_plot),
              list(NegNeg_N, NegPos_N, PosPos_N, PosNeg_N)))
}

# sum(sapply(speed_neurons$avg_both_asr_b, anyNA))

# Function to plot slopes as a function of trial outcome
slopes_by_outcome <- function(df, min_y = -3.5, max_y = 3.5){
  df[!(sapply(df$Avg_FiringRate_TryTrials, anyNA) | sapply(df$Avg_FiringRate_RunTrials, anyNA)),] %>% filter(lm_group_b == "Positive" | lm_group_b == "Negative") %>%
    select(unique_id, asr_b_o_rewarded_fit_slope, asr_b_try_fit_slope, asr_b_run_fit_slope) %>%
    rename(Hit = asr_b_o_rewarded_fit_slope,
           Try = asr_b_try_fit_slope,
           Run = asr_b_run_fit_slope) %>%
    mutate(unique_id = unlist(unique_id)) %>%
    pivot_longer(cols = c(Hit, Try, Run), names_to = "Outcome", values_to = "Slope", ) %>%
    ggplot(aes(x = fct_relevel(Outcome, "Hit", "Try", "Run"), y = Slope)) +
    coord_cartesian(ylim=c(min_y,max_y)) +
    geom_point() +
    geom_line(aes(group = unique_id, alpha = 0.5)) +
    geom_violin(aes(alpha = 0.5, fill = fct_relevel(Outcome, "Hit", "Try", "Run"))) +
    geom_hline(yintercept=0, linetype="dashed", color = "black") +
    labs(x = "Outcome", y = "Slope") +
    scale_fill_manual(values=c("grey","red", "blue")) +
    theme_classic() +
    theme(text = element_text(size=20),
          legend.position = "none")
}


# Function to plot offsets as a function of trial outcome
offsets_by_outcome <- function(df, min_y = -3.5, max_y = 3.5){
  df[!(sapply(df$Avg_FiringRate_TryTrials, anyNA) | sapply(df$Avg_FiringRate_RunTrials, anyNA)),] %>% filter(lm_group_b == "Positive" | lm_group_b == "Negative") %>%
    select(unique_id, predict_diff_hit, predict_diff_try, predict_diff_run) %>%
    rename(Hit = predict_diff_hit,
           Try = predict_diff_try,
           Run = predict_diff_run) %>%
    mutate(unique_id = unlist(unique_id)) %>%
    pivot_longer(cols = c(Hit, Try, Run), names_to = "Outcome", values_to = "Slope", ) %>%
    ggplot(aes(x = fct_relevel(Outcome, "Hit", "Try", "Run"), y = Slope)) +
    coord_cartesian(ylim=c(min_y,max_y)) +
    geom_point() +
    geom_line(aes(group = unique_id, alpha = 0.5)) +
    geom_violin(aes(alpha = 0.5, fill = fct_relevel(Outcome, "Hit", "Try", "Run"))) +
    geom_hline(yintercept=0, linetype="dashed", color = "black") +
    labs(x = "Outcome", y = "Offset") +
    scale_fill_manual(values=c("grey","red", "blue")) +
    theme_classic() +
    theme(text = element_text(size=20),
          legend.position = "none")
}

# One-way ANOVA to compare slopes by outcome
slopes_by_outcome_aov <- function(df) {
  df <- df[!(sapply(df$Avg_FiringRate_TryTrials, anyNA) | sapply(df$Avg_FiringRate_RunTrials, anyNA)),] %>% filter(lm_group_b == "Positive" | lm_group_b == "Negative") %>%
    select(unique_id, asr_b_o_rewarded_fit_slope, asr_b_try_fit_slope, asr_b_run_fit_slope) %>%
    rename(Hit = asr_b_o_rewarded_fit_slope,
           Try = asr_b_try_fit_slope,
           Run = asr_b_run_fit_slope) %>%
    mutate(unique_id = unlist(unique_id)) %>%
    pivot_longer(cols = c(Hit, Try, Run), names_to = "Outcome", values_to = "Slope", )
  #aov(Slope ~ as.factor(Outcome), data = df)
  aov(Slope ~ as.factor(Outcome) + Error(as.factor(unique_id)), data = df)
  #aov_2(df)
}

# To check AOV because F value was suspiciously low.  
aov_2 <- function(df){
  CF = (sum(df$Slope))^2/length(df$Slope)
  total.ss = sum(df$Slope^2)-CF
  between.ss = (sum(df$Slope[df$Outcome=="Try"])^2)/length(df$Slope[df$Outcome=="Try"]) +
    (sum(df$Slope[df$Outcome=="Run"])^2)/length(df$Slope[df$Outcome=="Run"]) +
    (sum(df$Slope[df$Outcome=="Hit"])^2)/length(df$Slope[df$Outcome=="Hit"]) - CF
  within.ss = total.ss - between.ss
  df.total = length(df$Slope) - 1
  df.between = length(unique(df$Outcome)) - 1
  df.within = df.total - df.between
  between.ms = between.ss/df.between
  within.ms = within.ss/df.within
  F.value = between.ms/within.ms
  list(CF, total.ss, between.ss, within.ss, df.total, df.between, df.within, between.ms, within.ms, F.value)
}

# One sample t-tests for slopes
slopes_by_outcome_t <- function(df) {
  df[!(sapply(df$Avg_FiringRate_TryTrials, anyNA) | sapply(df$Avg_FiringRate_RunTrials, anyNA)),] %>% filter(lm_group_b == "Positive" | lm_group_b == "Negative") %>%
    select(unique_id, asr_b_o_rewarded_fit_slope, asr_b_try_fit_slope, asr_b_run_fit_slope) %>%
    rename(Hit = asr_b_o_rewarded_fit_slope,
           Try = asr_b_try_fit_slope,
           Run = asr_b_run_fit_slope) %>%
    mutate(unique_id = unlist(unique_id)) %>%
    pivot_longer(cols = c(Hit, Try, Run), names_to = "Outcome", values_to = "Slope", ) %>%
    group_by(Outcome) %>%
    summarise(ttest = list(t.test(Slope, mu = 0)$p.value)) %>%
    unnest(cols = c(ttest))
}


# One-way ANOVA to compare offsets by outcome
offsets_by_outcome_aov <- function(df) {
  df <- df[!(sapply(df$Avg_FiringRate_TryTrials, anyNA) | sapply(df$Avg_FiringRate_RunTrials, anyNA)),] %>% filter(lm_group_b == "Positive" | lm_group_b == "Negative") %>%
    select(unique_id, predict_diff_hit, predict_diff_try, predict_diff_run) %>%
    rename(Hit = predict_diff_hit,
           Try = predict_diff_try,
           Run = predict_diff_run) %>%
    mutate(unique_id = unlist(unique_id)) %>%
    pivot_longer(cols = c(Hit, Try, Run), names_to = "Outcome", values_to = "Offset", )
aov <- aov(Offset ~ as.factor(Outcome) + Error(as.factor(unique_id)), data = df)
}

# One sample t tests for offsets
offsets_by_outcome_t <- function(df) {
  df[!(sapply(df$Avg_FiringRate_TryTrials, anyNA) | sapply(df$Avg_FiringRate_RunTrials, anyNA)),] %>% filter(lm_group_b == "Positive" | lm_group_b == "Negative") %>%
    select(unique_id, predict_diff_hit, predict_diff_try, predict_diff_run) %>%
    rename(Hit = predict_diff_hit,
           Try = predict_diff_try,
           Run = predict_diff_run) %>%
    mutate(unique_id = unlist(unique_id)) %>%
    pivot_longer(cols = c(Hit, Try, Run), names_to = "Outcome", values_to = "Offset", ) %>%
    group_by(Outcome) %>%
    summarise(ttest = list(t.test(Offset, mu = 0)$p.value)) %>%
    unnest(cols = c(ttest))
}



# -------------------------------------------------------------------------------- #

# Functions below here are initially in Figure 4 code.


# To plot firing rate slopes after the reward zone on beaconed vs probe trials
b_vs_p_h_slope_plot <- function(df){
  ggplot() + 
    geom_point(data=subset(df, track_category == "pospos" | track_category == "negneg"),
               aes(x = as.numeric(unlist(asr_b_h_rewarded_fit_slope)), 
                   y = as.numeric(unlist(asr_p_h_rewarded_fit_slope)), 
                   color=factor(unlist(lm_group_b))), alpha=0.8) +
    geom_point(data=subset(df, track_category == "posneg" | track_category == "negpos"),
               aes(x = as.numeric(unlist(asr_b_h_rewarded_fit_slope)), 
                   y = as.numeric(unlist(asr_p_h_rewarded_fit_slope)), 
                   color=factor(unlist(lm_group_b))), shape=2, alpha=0.8) +
    geom_point(data=subset(df, track_category == "posnon" | track_category == "negnon"),
               aes(x = as.numeric(unlist(asr_b_h_rewarded_fit_slope)), 
                   y = as.numeric(unlist(asr_p_h_rewarded_fit_slope)), 
                   color=factor(unlist(lm_group_b))), shape=3, alpha=0.8) + 
    geom_smooth(data=subset(df, track_category != "None"),aes(x=asr_b_h_rewarded_fit_slope, y=asr_p_h_rewarded_fit_slope), method = "lm", se = FALSE, color ="red", size = 0.5, linetype="dashed") +
    geom_abline(intercept = 0, slope = 1, colour = "grey", linetype = "dashed") +
    xlab("Beaconed slope") +
    ylab("Probe slope") +
    theme_classic() +
    scale_color_manual(values=c("violetred2", "chartreuse3", "grey")) +
    theme(axis.text.x = element_text(size=17),
          axis.text.y = element_text(size=17),
          legend.position="bottom", 
          legend.title = element_blank(),
          text = element_text(size=16), 
          legend.text=element_text(size=16), 
          axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))) 
}


# To plot firing rate slopes before the reward zone on beaconed vs probe trials
b_vs_p_o_slope_plot <- function(df){
  ggplot() + 
    geom_point(data=subset(df, track_category == "pospos" | track_category == "negneg"),
               aes(x = as.numeric(unlist(asr_b_o_rewarded_fit_slope)), 
                   y = as.numeric(unlist(asr_p_o_rewarded_fit_slope)), 
                   color=factor(unlist(lm_group_b))), alpha=0.8) +
    geom_point(data=subset(df, track_category == "posneg" | track_category == "negpos"),
               aes(x = as.numeric(unlist(asr_b_o_rewarded_fit_slope)), 
                   y = as.numeric(unlist(asr_p_o_rewarded_fit_slope)), 
                   color=factor(unlist(lm_group_b))), shape=2, alpha=0.8) +
    geom_point(data=subset(df, track_category == "posnon" | track_category == "negnon"),
               aes(x = as.numeric(unlist(asr_b_o_rewarded_fit_slope)), 
                   y = as.numeric(unlist(asr_p_o_rewarded_fit_slope)), 
                   color=factor(unlist(lm_group_b))), shape=3, alpha=0.8) + 
    geom_smooth(data=subset(df, track_category != "None"),aes(x=asr_b_o_rewarded_fit_slope, y=asr_p_o_rewarded_fit_slope), method = "lm", se = FALSE, color ="red", size = 0.5, linetype="dashed") +
    geom_abline(intercept = 0, slope = 1, colour = "grey", linetype = "dashed") +
    xlab("Beaconed slope") +
    ylab("Probe slope") +
    theme_classic() +
    scale_color_manual(values=c("violetred2", "chartreuse3", "grey")) +
    theme(axis.text.x = element_text(size=17),
          axis.text.y = element_text(size=17),
          legend.position="bottom", 
          legend.title = element_blank(),
          text = element_text(size=16), 
          legend.text=element_text(size=16), 
          axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))) 
}


# To plot firing rate slopes before the reward zone on beaconed vs probe trials
b_vs_p_offset_plot <- function(df){
  ggplot() + 
    geom_point(data=subset(df, track_category == "pospos" | track_category == "negneg"),
               aes(x = predict_diff, 
                   y = predict_diff_p, 
                   color=factor(unlist(lm_group_b))), alpha=0.8) +
    geom_point(data=subset(df, track_category == "posneg" | track_category == "negpos"),
               aes(x = predict_diff, 
                   y = predict_diff_p, 
                   color=factor(unlist(lm_group_b))), shape=2, alpha=0.8) +
    geom_point(data=subset(df, track_category == "posnon" | track_category == "negnon"),
               aes(x = predict_diff, 
                   y = predict_diff_p, 
                   color=factor(unlist(lm_group_b))), shape=3, alpha=0.8) + 
    geom_smooth(data=subset(df, track_category != "None"),aes(x=predict_diff, y=predict_diff_p), method = "lm", se = FALSE, color ="red", size = 0.5, linetype="dashed") +
    geom_abline(intercept = 0, slope = 1, colour = "grey", linetype = "dashed") +
    xlab("Beaconed offset") +
    ylab("Probe offset") +
    theme_classic() +
    scale_color_manual(values=c("violetred2", "chartreuse3", "grey")) +
    theme(axis.text.x = element_text(size=17),
          axis.text.y = element_text(size=17),
          legend.position="bottom", 
          legend.title = element_blank(),
          text = element_text(size=16), 
          legend.text=element_text(size=16), 
          axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))) 
}



# Helper function to format data for mean_SEM_plots_comp
extract_cols_for_plot <- function(df, bin = 200){
  df <- tibble(Position = rep(-29.5:(-29.5+199), times=nrow(df)), 
               Rates = unlist(df$normalised_rates_smoothed),
               Rates_c = unlist(df$normalised_rates_p_smoothed),
               Outbound_beaconed_b = rep(df$lm_group_b, each=bin), 
               Homebound_beaconed_b = rep(df$lm_group_b_h, each=bin), 
               Outbound_beaconed_p = rep(df$lm_group_p, each=bin), 
               Homebound_beaconed_p = rep(df$lm_group_p_h, each=bin)) 
}


# Function to add extra trace in Rates_c to a mean_SEM_plot.
# The function mean_SEM_plots was first used in Figure 1.
# The colimns mean and sem generate a blue line, with mean_c and sem_c generating a black line.

mean_SEM_plots_comp_prep <- function(df) {
  df <- df %>% dplyr::summarise(mean_r = mean(Rates, na.rm = TRUE),
                                sem_r = std.error(Rates, na.rm = TRUE),
                                mean_c = mean(Rates_c, na.rm = TRUE),
                                sem_c = std.error(Rates_c, na.rm = TRUE))
}

mean_SEM_plots_comp <- function(df, colour1 = "black", colour2 = "blue"){
  plot <- mean_SEM_plots(df, colour1)
  
  plot +
    geom_ribbon(aes(x=Position, y=mean_c, ymin = mean_c - sem_c, ymax = mean_c + sem_c), fill = colour2, alpha=0.2) +
    geom_line(aes(y=mean_c, x=Position), color = colour2) 
}


# OB_b, HB_b, OB_p, HB_p are strings that define the slope for each track segment and trial type 
plot_beaconed_vs_probe_all <- function(df, OB_b, HB_b, OB_p, HB_p) {
  df  %>%
    extract_cols_for_plot() %>%
    filter(Outbound_beaconed_b == OB_b &
             Homebound_beaconed_b == HB_b) %>%
    filter(Outbound_beaconed_p == OB_p &
             Homebound_beaconed_p == HB_p) %>%
    group_by(Position) %>%
    mean_SEM_plots_comp_prep() %>%
    mean_SEM_plots_comp()
}

plot_beaconed_vs_probe <- function(df, OB_b, HB_b) {
  df  %>%
    extract_cols_for_plot() %>%
    filter(Outbound_beaconed_b == OB_b &
             Homebound_beaconed_b == HB_b) %>%
    group_by(Position) %>%
    mean_SEM_plots_comp_prep() %>%
    mean_SEM_plots_comp()
}



# Functions to make families of plots comparing beaconed and probe trials
# Version that selects neurons based on outbound and homebound classifications
comp_beacon_probe_rate_plots_all <- function(df){
  a <- df %>% plot_beaconed_vs_probe_all("Negative", "Negative", "Negative", "Negative")
  
  b <- df %>% plot_beaconed_vs_probe_all("Negative", "Positive", "Negative", "Positive")
  
  c <- df %>% plot_beaconed_vs_probe_all("Positive", "Positive", "Positive", "Positive")
  
  d <- df %>% plot_beaconed_vs_probe_all("Positive", "Negative", "Positive", "Negative")
  
  return(list(a, b, c, d))
}

# Version that selects neurons based only on outbound classifications
comp_beacon_probe_rate_plots <- function(df){
  a <- df %>% plot_beaconed_vs_probe("Negative", "Negative")
  
  b <- df %>% plot_beaconed_vs_probe("Negative", "Positive")
  
  c <- df %>% plot_beaconed_vs_probe("Positive", "Positive")
  
  d <- df %>% plot_beaconed_vs_probe("Positive", "Negative")
  
  return(list(a, b, c, d))
}



## ----------------------------------------------------------##
## ----------------------------------------------------------##

# Older functions. Not clear if they are used. Delete?!

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