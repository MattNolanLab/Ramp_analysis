

car_pos_b <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(df[[1]]), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 110 & Position <= 170 & Speed > 3 & Types == 0) 
  if(length(df) == 1) 
    return(NA) 
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[1])
}

car_speed_b <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(df[[1]]), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 110 & Position <= 170 & Speed > 3 & Types == 0) 
  if(length(df) == 1) 
    return(NA) 
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[2])
}

car_accel_b <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(df[[1]]), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 110 & Position <= 170 & Speed > 3 & Types == 0)
  if(length(df) == 1) 
    return(NA) 
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[3])
}


car_pos_nb <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(df[[1]]), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 110 & Position <= 170 & Speed > 3 & Types == 1) 
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  #print(nrow(df))
  #print(df)
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[1])
}

car_speed_nb <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(df[[1]]), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 110 & Position <= 170 & Speed > 3 & Types == 1) 
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[2])
}

car_accel_nb <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(df[[1]]), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 110 & Position <= 170 & Speed > 3 & Types == 1) 
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[3])
}



car_pos_p <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(df[[1]]), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 110 & Position <= 170 & Speed > 3 & Types == 2)
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA)  
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[1])
}

car_speed_p <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(df[[1]]), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 110 & Position <= 170 & Speed > 3 & Types == 2)
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA) 
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[2])
}

car_accel_p <- function(df){
  if(length(df) == 1) 
    return(NA) 
  df <- tibble(Rates = as.numeric(df[[1]]), Position = as.numeric(Re(df[[2]])), Acceleration = as.numeric(Re(df[[3]])), Speed = as.numeric(Re(df[[4]])), Trials = as.factor(df[[5]]), Types = as.factor(df[[6]]))
  df <- df %>% 
    subset(Position >= 110 & Position <= 170 & Speed > 3 & Types == 2)
  if(length(df) == 1 | nrow(df) < 3) 
    return(NA)  
  df_int <- lmer(Rates ~ Position + Speed + Acceleration + (1|Trials), data = df, na.action=na.exclude, REML=FALSE)
  prtAnova <- car::Anova(df_int) 
  return(prtAnova$"Pr(>Chisq)"[3])
}


