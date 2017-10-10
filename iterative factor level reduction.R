install.packages("tidyverse")
install.packages("readr")
install.packages("forcats")
install.packages("caret")
install.packages("pryr")
install.packages("modelr")
install.packages("stringr")
install.packages("C50")
install.packages("lazyeval")
install.packages("lubridate")
install.packages("dummies")

setwd("F:\\STUDY\\Study@Uconn\\PMG\\Group Project 2 fall 2016\\R code")

library(tidyverse)
library(readr)
library(forcats)
library(caret)
library(pryr)
library(modelr)
library(stringr)
library(C50)
library(lazyeval)
library(lubridate)
library(lattice)
library(dummies)
#unload clashing packages
pkgs <- search()
if("package:MASS" %in% pkgs){
  detach("package:MASS")
}


sess <- read_csv("../sessions.csv", na = ".")
sess %>% count
sess %>% summary()
save.image("s1.RData")
load("s1.RData")
sess1 <- sess %>% filter(user_id != "" & action != "" & action_type != "" & action_detail != "" & device_type != "" 
                         & secs_elapsed != "")
sess1 %>% count(action)

tr <- read_csv(file = "../train_users_2.csv", na = ".")
tr %>% summary()
tr1 <- tr %>% filter(id != "")

sess1 <- tr1 %>%
  inner_join(sess1, c("id" = "user_id")) %>%
  mutate(is_booking = fct_collapse(country_destination,
                                   N = c("NDF"),
                                   Y = c("AU", "CA", "DE", "ES", "FR", "GB", "IT", "NL", "other", "PT", "US")
  ))

sess1 %>% count(is_booking)		#baseline factor level ratio

set.seed(3457)
trainIndex <- createDataPartition(sess1$is_booking, p = .5, 
                                  list = FALSE, 
                                  times = 1)
sess1_tv <- sess1[trainIndex, ]
sess1_ts <- sess1[-trainIndex, ]

#down-sampling
sess_tmp <- sess1_tv %>% select(-is_booking) 
sess1_tv <- downSample(x = sess_tmp, y = sess1_tv$is_booking, yname = "is_booking")
sess1_tv %>% count(is_booking)

computeC5.0ModelOutput <- function(df, class){
  
  nzv <- nearZeroVar(df)			#drop columns with zero/near zero variance
  #print(nzv)
  if(length(nzv) != 0){
    df <- df[, -nzv]  
  }
  #print(colnames(df))
  
  if(ncol(df) == 2){
    print("Only 1 significant level remaining. Cannot run model using constants!")
    return(-1)
  }
  
  descrCor <- cor(df[,-ncol(df)])
  print("Correlation summary:")
  print(summary(descrCor[upper.tri(descrCor)]))
  #print("Correlation matrix:")
  #print(descrCor)
  
  highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
  #print(highlyCorDescr)
  if(length(highlyCorDescr) != 0){
    df <- df[,-highlyCorDescr]
    
    print("Correlation summary after dropping flagged predictors:")
    descrCor2 <- cor(df[,-ncol(df)])
    print(summary(descrCor2[upper.tri(descrCor2)]))
  }
  #print(dim(df))
  
  set.seed(3458)
  trainIndex <- createDataPartition(df[[class]], p = .6, 
                                    list = FALSE, 
                                    times = 1)
  df_tr <- df[trainIndex, ]
  df_vl <- df[-trainIndex, ]
  df_tr %>% count(df_tr[[class]])		#verify balanced factor level ratio
  #object_size(df_tr)
  
  model <- runC5.0Model(df_tr, class)
  print(model)
  
  displayCM(model, df_tr, class)
  displayCM(model, df_vl, class)
  
  res <- displayPredAndImpPred(model)
  return(res)
}

runC5.0Model <- function(df, class){
  
  set.seed(826)
  ctrl <- trainControl(method = "none",
                       classProbs = TRUE, 
                       summaryFunction = twoClassSummary)
  
  #print(length(df[[class]]))
  df <- as.data.frame(df)
  #print(ncol(df))
  model <- train(df[, -ncol(df), drop = FALSE], df[[ncol(df)]],
                 method = "C5.0Tree",
                 trControl = ctrl,
                 ## Below options are method's pass through
                 #tuneLength = 1,
                 #metric = "ROC", 
                 preProc = c("center", "scale")
                 #preProc = c("center", "scale", "pca")			
  )
  return(model)
}

displayCM <- function(model, df, class){
  modelLabels <- predict(model, df[, -ncol(df)])
  print(confusionMatrix(data = modelLabels, reference = df[[class]]))
}

displayPredAndImpPred <- function(model){
  #print(predictors(model))
  print(varImp(model))
  return(varImp(model))
}

prepDataAndAnalyse <- function(df, target_name, grp_col_name, out_filename, id_name = NA, agg_col_name = "def_agg", agg_fn = "n", agg_fn_args = NA){
  if(!is.na(id_name)){
    if(is.na(agg_fn_args)){
      eval_expr1 <- interp(~ x(), x = as.name(agg_fn))
    }
    else{
      eval_expr1 <- interp(~ x(y), x = as.name(agg_fn), y = as.name(agg_fn_args))
    }
    dots <- list(eval_expr1)
    eval_expr2 <- interp(~ starts_with(x), x = grp_col_name)
    df <- df %>% 
      group_by_(id_name, grp_col_name, target_name) %>%
      summarise_(.dots = setNames(dots, c(agg_col_name))) %>%	
      select_(id_name, grp_col_name, agg_col_name, target_name) %>%
      spread_(key = grp_col_name, value = agg_col_name, sep = "_") %>%
      ungroup() %>%
      select_(eval_expr2, target_name)
    
    df[is.na(df)] <- 0        #patch NAs produced by spread()
    #colnames(df)
  }
  else{
    df <- df %>% select_(grp_col_name, target_name)
    eval_expr1 <- interp(~ x - 1, x = as.name(grp_col_name))
    mm <- model_matrix(df, eval_expr1)
    df <- cbind(mm, df[[target_name]])
    colnames(df)[ncol(df)] <- target_name     #patch corrupted target column name
    regex <- paste('^(', grp_col_name, ')(.+)$', sep = "")
    colnames(df) <- gsub(regex, '\\1_\\2', colnames(df))
  }
  
  sink(out_filename)
  res <- computeC5.0ModelOutput(df, target_name)
  sink()
  return(res)
}


imp_pred <- prepDataAndAnalyse(sess1_tv, "is_booking", "action", "model_output_action_count.txt", "id", "count_action")

prepDataAndAnalyse(sess1_tv, "is_booking", "action", "model_output_action_secs.txt", "id", "total_action_secs", "sum", "secs_elapsed")

imp_pred <- prepDataAndAnalyse(sess1_tv, "is_booking", "action_detail", "model_output_detail_count.txt", "id", "count_detail")

prepDataAndAnalyse(sess1_tv, "is_booking", "action_detail", "model_output_user_detail_secs.txt", "id", "total_detail_secs", "sum", "secs_elapsed")

prepDataAndAnalyse(sess1_tv, "is_booking", "action_type", "model_output_type_count.txt", "id", "count_type")

prepDataAndAnalyse(sess1_tv, "is_booking", "device_type", "model_output_device_count.txt", "id", "count_device")

prepDataAndAnalyse(sess1_tv, "is_booking", "action_detail", "model_output_detail_count_pca.txt", "id", "count_detail")

#adding secs_elapsed in model reduces overall accuracy. So we won't consider it further
#device_type has only 1 significant level, so cannot be used for modeling as it will behave as a constant
#action_type has an 'unknown' level which is picked as significant factor. So we won't use this column as its ambiguous
#no significant impact on model accuracy using PCA. Hence we can avoid PCA as it affects interpretability
#we can choose action_detail for it has best accuracy and factor level importance distribution


reduceFactorLevels <- function(df, target_name, grp_col_name, out_filename, id_name = NA, agg_col_name = "def_agg"){
  
  regex_str = paste("^", grp_col_name, "_", sep = "")
  while(TRUE){
    if(nlevels(df[[grp_col_name]]) == 2){
      print("Factor level reduction not possible further as only 2 levels exist")
      print(paste("Resultant levels for '", grp_col_name,"' are:", sep = ""))
      print(levels(df[[grp_col_name]]))
      break
    }
    imp_pred <- prepDataAndAnalyse(df, target_name, grp_col_name, out_filename, id_name, agg_col_name)
    #print(str(imp_pred))
    if(identical(imp_pred, -1)){
      print("Error during modeling setup. Likely that predictors don't have sufficient variance")
      return(-1)
    }
    imp_pred[["importance"]]$Predictors <- str_replace(rownames(imp_pred[["importance"]]), regex_str, "")
    #print(imp_pred[[1]]$Predictors)
    sig_pred <- imp_pred[[1]][which(imp_pred[[1]]$Overall >= 35.00), ]
    all_pred <- levels(as.factor(df[[grp_col_name]]))
    insig_pred <- setdiff(all_pred, sig_pred$Predictors)
    print(paste("#less-significant factor levels are:", length(insig_pred)))
    
    if(length(insig_pred)== 1){
      print(paste("Least significant factor level:", insig_pred))
      print("Iterative factor level reduction finished as only 1 insignificant level exists")
      print(paste("Resultant levels for '", grp_col_name,"' are:", sep = ""))
      print(levels(df[[grp_col_name]]))
      break
    }
    else{
      #collapse less significant levels of concerned column into 'Other'
      eval_expr1 <- interp(~ fct_collapse(x, Other = insig_pred), x = as.name(grp_col_name))
      dots = list(eval_expr1)
      df <- df %>% mutate_(.dots = setNames(dots, c(grp_col_name)))
    }
  }
  return(df)
}

sess2_tv <- sess1_tv
sess2_tv <- reduceFactorLevels(sess2_tv, "is_booking", "action", "model_output_action_count2.txt", "id", "count_action")
sess2_tv <- reduceFactorLevels(sess2_tv, "is_booking", "action_detail", "model_output_detail_count2.txt", "id", "count_detail")

summary(sess2_tv$action)
summary(sess2_tv$action_detail)

colnames(sess2_tv)

user_tv <- sess2_tv %>% select(-id, -action_type, -device_type, -secs_elapsed, -country_destination, -date_first_booking)
colnames(user_tv)
summary(user_tv$timestamp_first_active)

#treat numeric column as string and use first 8 characters to form Date object
user_tv$timestamp_first_active <- as.character(as.numeric(user_tv$timestamp_first_active))
str(user_tv$timestamp_first_active)    
user_tv$timestamp_first_active <-
  parse_date_time(user_tv$timestamp_first_active, orders = "YmdHMS")
str(user_tv$timestamp_first_active)     #POSIXct

user_tv$date_account_created <- parse_date_time(user_tv$date_account_created, orders = "Ymd")
str(user_tv$date_account_created)       #POSIXct

user_tv$date_diff <- as.Date(user_tv$timestamp_first_active) - as.Date(user_tv$date_account_created)
summary(as.numeric(user_tv$date_diff))

#since timestamp_first_active and date_account_created are equal, we drop one of these
user_tv <- user_tv %>% select(-date_account_created)
colnames(user_tv)    

user_tv$year <- year(user_tv$timestamp_first_active)
user_tv$month <- month(user_tv$timestamp_first_active)
user_tv$day <- day(user_tv$timestamp_first_active)
user_tv$hour <- hour(user_tv$timestamp_first_active)
summary(user_tv$year)
summary(user_tv$month)
summary(user_tv$day)
summary(user_tv$hour)

user_tv <- user_tv %>% select(-timestamp_first_active)

save.image("s2.RData")
load("s2.RData")

dim(user_tv)  

summary(factor(user_tv$gender))
histogram(~ is_booking | gender, user_tv)
user1_tv <- reduceFactorLevels(user_tv, "is_booking", "gender", "model_output_gender.txt")
histogram(~ is_booking | gender, user1_tv)
#user1_tv %>% ggplot() + geom_bar(mapping = aes(is_booking)) + facet_wrap(~gender)
#xyplot(~ month | , data = user_tv)
#histogram(~ month | is_booking, user_tv)

remove(sess, sess_tmp, sess1, sess1_tv, sess2_tv, tr, tr1, mm, mm2)
save.image("s3.RData")
load("s3.RData")

summary(user1_tv$age)
histogram(~ age, user1_tv)
total_count <- user1_tv %>% count()
user1_tmp <- user1_tv %>% filter(is.na(age) | (age >= 18 & age <= 105))
histogram(~ age, user1_tmp)
nrow(user1_tv)    #2146090 rows
nrow(user1_tmp)   #2138062 rows
user1_tmp$age <- factor(user1_tmp$age)
levels(user1_tmp$age) <- c(levels(user1_tmp$age), -1)
summary(user1_tmp$age)  #NAs are preserved
user1_tmp$age[is.na(user1_tmp$age)] <- -1   #replace missing ages with -1
summary(user1_tmp$age)  #NAs are replaced with new level: -1
user1_tmp <- reduceFactorLevels(user1_tmp, "is_booking", "age", "model_output_age.txt")
#results show that individually none of the levels of 'age' are significant
