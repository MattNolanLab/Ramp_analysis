

#1. Function to calculate standardized coefficients for a LMER
#https://stackoverflow.com/questions/25142901/standardized-coefficients-for-lmer-model 


stdCoef.merMod <- function(object) {
  sdy <- sd(getME(object,"y"))
  sdx <- apply(getME(object,"X"), 2, sd)
  sc <- fixef(object)*sdx/sdy
  se.fixef <- coef(summary(object))[,"Std. Error"]
  se <- se.fixef*sdx/sdy
  return(data.frame(stdcoef=sc, stdse=se))
}



#2. Function to calculate and extract standardized coefficients
coef_ratio_pos <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(Re(df[[5]])), Types = as.factor(Re(df[[6]])))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 0)
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lme4::lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude)
  # standardize parameters first
  mod <- stdCoef.merMod(df_int) #print(effectsize::standardize_parameters(df_int)) --> alternative way of standardising the coefficients
  #mod <- sjstats::std_beta(df_int, ci.lvl = 0.95)
  pos <- mod[2,1]
  speed <- mod[3,1]
  accel <- mod[4,1]
  return(pos)
}

coef_ratio_speed <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(Re(df[[5]])), Types = as.factor(Re(df[[6]])))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 0)
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lme4::lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude)
  # standardize parameters first
  mod <- stdCoef.merMod(df_int) #print(effectsize::standardize_parameters(df_int)) --> alternative way of standardising the coefficients
  pos <- mod[2,1]
  speed <- mod[3,1]
  accel <- mod[4,1]
  return(speed)
}

coef_ratio_accel <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(Re(df[[1]])), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(Re(df[[5]])), Types = as.factor(Re(df[[6]])))
  df <- df %>% 
    subset(Position >= 30 & Position <= 90 & Speed > 3 & Types == 0)
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lme4::lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude)
  # standardize parameters first
  mod <- stdCoef.merMod(df_int) #print(effectsize::standardize_parameters(df_int)) --> alternative way of standardising the coefficients
  pos <- mod[2,1]
  speed <- mod[3,1]
  accel <- mod[4,1]
  return(accel)
}