
#function2 - shuffled and real : beaconed trials
return_b_shuffle_lm_results <- function(df){
  tibble(
    Trial_type = factor(c(rep(c("Real","Shuffled"), each =nrow(df)))),
    Slopes = c(df$asr_b_slope_o, df$shuff_asr_b_slope_o), 
    r2 = c(df$asr_b_r2_o, df$shuff_asr_b_r2_o), 
    id = c(df$cluster_id, df$cluster_id))
}


## ----------------------------------------------------------##



lm_helper <- function(df, bins){
  if(all(is.na(df))) 
    return(0)
  df_mod <- lm(Rates ~ Position, data = df[bins,], na.action=na.exclude)
}

## ----------------------------------------------------------##


#Next we want to perform LM on each cluster to analyse firing rate verses location.
#- Make a new columns in dataframe for output of linear model
#- Fit the model from average rates versus location for each cluster
#- Then put the results into new columns of the dataframe


lm_analysis <- function(df, spike_rate_col, startbin = 30, endbin = 90) {
  spike_rate_col <- enquo(spike_rate_col)
  out_name <- sym(paste0(quo_name(spike_rate_col)))
  sr_unnest_name <- sym(paste0(quo_name(spike_rate_col), "_unnest"))
  fit_name <- sym(paste0(quo_name(out_name), "_fit"))
  glance_name <- sym(paste0(quo_name(out_name), "_glance"))
  r2_name <- sym(paste0(quo_name(out_name), "_r2_o"))
  Pval_name <- sym(paste0(quo_name(out_name), "_Pval_o"))
  slope_name <- sym(paste0(quo_name(out_name), "_slope_o"))
  intercept_name <- sym(paste0(quo_name(out_name), "_intercept_o"))
  df <- df %>%
    mutate(!!fit_name := map(!!spike_rate_col, bins=c(startbin:endbin), lm_helper),
           !!glance_name := map(!!fit_name, glance),
           !!r2_name := map_dbl(!!glance_name, ~.$r.squared),
           !!Pval_name := map_dbl(!!glance_name, ~.$p.value),
           !!slope_name := map_dbl(!!fit_name, ~.$coefficients[2]),
           !!intercept_name := map_dbl(!!fit_name, ~.$coefficients[1]))
}



## ----------------------------------------------------------##


select_final_lm_result <- function(pval, r2, slope, max_r2, max_slope, min_slope){
  if (is.na(r2) | is.na(slope)| is.na(pval) ) {
    return( "None" )
  } else if( r2 > max_r2 & slope > max_slope & pval < 0.05) {
    return( "Positive" )
  } else if( r2 > max_r2 & slope < min_slope & pval < 0.05) {
    return( "Negative" )
  } else if( r2 > max_r2 & slope > min_slope & slope < max_slope){
    return("None")
    } else {
    return("None")
  }
}


## ----------------------------------------------------------##



sum_rates <- function(df){
  if(all(is.na(df))) 
    return(0)
  x = sum(df$Rates)
}


## ----------------------------------------------------------##



car_pos_b <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 0)
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lme4::lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[1])
}

car_speed_b <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 0)
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lme4::lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[2])
}

car_accel_b <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 0) 
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lme4::lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[3])
}


car_pos_nb <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 1) 
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[1])
  
}

car_speed_nb <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 1) 
  if(length(df) == 1 | nrow(df) < 3) 
      return(NA) 
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[2])
}

car_accel_nb <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 1) 
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[3])
}


car_pos_p <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 2)
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA)  
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[1])
}

car_speed_p <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 2) 
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[2])
}

car_accel_p <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 2) 
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA)  
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[3])
}


## ----------------------------------------------------------##



model_comparison <- function(null_pos, null_speed, null_accel){
  pval <- 0.001
  if( is.na(null_pos) & is.na(null_accel)) {
    return( "None" )
    
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
