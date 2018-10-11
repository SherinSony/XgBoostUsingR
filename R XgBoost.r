
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(ggplot2)
library(caret)
library(dplyr)
library(e1071)
library(mlr) 
library(ROCR)
library(verification)
library(data.table)
library(car)
library(stringr)
library(readr)
library(ResourceSelection)
library(Matrix)
library(gains)
library(irr)
library(MLmetrics)
library(ggmap)
library(splitstackshape)
library(party)
library(caTools)
library(xgboost)

set.seed(100)	
setwd("D:\\mypath")
df<-read.csv("train.csv")
table(df$is_promo) #skewed target class - upsampling/downsampling to be done
#0     1 
#50140  4668 
summary(df)

#note: we can do an ifelse only on a string, if we do it on a factor, it gets auto converted into a number !
str(df)
df$region=as.factor(df$region)
df$kpi=as.factor(df$kpi)

df$edu1= as.character(df$edu) #first convert to a string to replace "" as "missing"
df$edu2= ifelse(df$edu1=="", "Missing",df$edu1)
df$edu1= as.factor(df$edu2)
summary(df)
df$edu= df$edu1
df$edu1=NULL
df$edu2=NULL
df$Filename="TR"
df$awards=as.factor(df$awards)

#------------ test file -----------------

ts=read.csv("test.csv")
ts$region=as.factor(ts$region)
ts$kpi=as.factor(ts$kpi)
ts$edu1= as.character(ts$edu) #first convert to a string to replace"" as "missing"
ts$edu2= ifelse(ts$edu1=="", "Missing",ts$edu1)
ts$edu1= as.factor(ts$edu2)
ts$edu= ts$edu1
ts$edu1=NULL
ts$edu2=NULL
ts$Filename="TS"
ts$awards=as.factor(ts$awards)


#----------one-hot encoding for categorical cols column----------
#combine test and train first
colnames(df)
comb=rbind(df[,-c(14)], ts)
dim(comb)
dim(df)
dim(ts)

vars=setdiff(colnames(comb), c("yrsinservice","prevyrrating","age","no_of_trainings","is_promo", "employee_id","Filename","trainscore"))
f <- paste('~', paste(vars, collapse = ' + '))
encoded<-caret::dummyVars(as.formula(f),comb)
comb_enc<-predict(encoded,comb)
comb2<-data.frame(comb_enc,comb)

#remove cols which were parent-cols for encoding
vars2=setdiff(colnames(comb2), c("department","region","edu","gender","recruitedvia","kpi","awards"))
comb3=comb2[,vars2]
str(comb3)
summary(comb3)
#replace missing values in prev yr ratings
comb3$prevyrrating=ifelse(is.na(comb3$prevyrrating),-1,comb3$prevyrrating)

#--------Normalization of the numeric cols ------------
vars_norm=c("age","prevyrrating","yrsinservice","trainscore","no_of_trainings")
comb3_norm=comb3[,which(colnames(comb3)%in%vars_norm)]  #subset by column names
preprocessParams <- preProcess(comb3_norm, method=c("range"))
transformed <- predict(preprocessParams, comb3_norm)


#join back with the original data in comb3
vars_norm2=setdiff(colnames(comb3), vars_norm)
comb33=cbind(comb3[,vars_norm2],transformed)
summary(comb33)
#save back into comb3
comb3=comb33


# ----------- Train Test Split ------------------

#now split back into test and train and merge train back with the is_prommoted col
tr1=comb3 %>% filter(Filename=="TR")
tr=cbind(tr1, is_promo=df$is_promo)  #rename the default dataframe.colummnname naming standards of cbind()
tst=comb3 %>% filter(Filename=="TS")
tst$Filename=NULL
tr$Filename=NULL

#--------------- split into test and train -------------------

index <- createDataPartition(tr$is_promo, p=0.70, list=FALSE)
trainSet <- tr[ index,]
testSet <- tr[-index,]

trainSet$is_promo=as.factor(trainSet$is_promo)
testSet$is_promo=as.factor(testSet$is_promo)

#---------------------------MODELING-----------------------------------------
#https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
str(trainSet)
trControl = caret::trainControl(sampling = "up", #,seeds = 100, 
method = 'cv',
  number = 2,
  classProbs = TRUE,
  verboseIter = TRUE,
  allowParallel = TRUE)

tuneGridXGB <- expand.grid(
  nrounds=c(100),
  max_depth = c(4,6,8, 10),
  eta = c(0.05, 0.1,0.02, 0.3),
  gamma = c(0.01),
  colsample_bytree = c(0.4,0.6,0.8,1.0),
  subsample = c(0.5,0.75, 1.0 ),
  min_child_weight = c(0))

start <- Sys.time()
table(trainSet$is_promo)

# train the xgboost learner

xgbmod <- caret::train(
  x = data.matrix(trainSet[,vars_xgb]),
  y = factor(ifelse(trainSet$is_promo==0, "Zero", "One")),
  method = 'xgbTree',
  metric = 'auc',
  trControl = trControl,
  tuneGrid = tuneGridXGB)

end <- Sys.time()

#--------------------PREDICTION FOR VALIDATION TEST -------------------------
pred_xgb_cv <- predict(xgbmod, newdata = testSet[,vars_xgb], type = "prob")

y_test =factor(ifelse(testSet$is_promo==0, "Zero", "One"))
levels(y_test)
y_test_bkp=y_test
y_test=y_test_bkp
levels(y_test) <- c("1", "0")
y_test_raw <- as.numeric(levels(y_test))[y_test]

class(pred_xgb_cv)
roc.plot(y_test_raw, pred_xgb_cv$One, plot.thres = c(0.02, 0.03, 0.04, 0.05))
pred_class<-ifelse(pred_xgb_cv$One>=0.09,1,0)

MLmetrics::ConfusionMatrix(y_pred =pred_class, y_true = testSet$is_promo ) 
cm=confusionMatrix(table(testSet$is_promo, pred_class),positive="1")
as.data.frame(cm$byClass)["F1",1]

#--------------------PREDICTION FOR FINAL TEST -------------------------

vars_xgb_finaltest_cv2=setdiff(colnames(tst), c("employee_id"))
preds_final_cv2 <- predict(xgbmod, newdata = tst[,vars_xgb_finaltest_cv2], type = "prob")
pred_classfinaltst_cv2<-ifelse(preds_final_cv2$One>=0.09,1,0)
tstfinalresult_cv2=as.data.frame(cbind(employee_id=tst$employee_id,is_promo=pred_classfinaltst_cv2))
write.csv(tstfinalresult_cv2,file = "Submission3_0.09.csv", row.names=FALSE)
