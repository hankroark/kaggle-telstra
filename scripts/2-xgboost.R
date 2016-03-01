library(caret)
library(xgboost)
library(readr)
library(dplyr)
library(tidyr)

train <- read_csv("data-postprep/train.csv")
test  <- read_csv("data-postprep/test.csv")

# integer encode the locations
l<-unique(train$derivlocation)
train$derivlocation <- as.numeric(factor(train$derivlocation, levels=l))
test$derivlocation <- as.numeric(factor(test$derivlocation, levels=l))

y <- "fault_severity"
# x <- setdiff(names(train), c(y,"derivlocation"))
x <- setdiff(names(train), c(y, names(train)[51:381]))
num.classes <- length(unique(train$fault_severity))

x.matrix <- as.matrix( sapply(train[,x], as.numeric) )
y.matrix <- as.matrix( sapply(train[,y], as.numeric) )
test.matrix <- as.matrix( sapply(test[,x], as.numeric) )
xgb_params_1 = list(
  objective = "multi:softprob",
  num_class = num.classes,
  eta = 0.01, # learning rate
  max.depth = 5, # max tree depth
  gamma = 0.0625,
  subsample=0.7,
  colsample_bytree=1,
  eval_metric = "mlogloss" # evaluation/loss metric
)

# cross-validate xgboost to get the accurate measure of error
xgb_cv_1 = xgb.cv(params = xgb_params_1,
                  data = x.matrix,
                  label = y.matrix,
                  nrounds = 10000, 
                  nfold = 5,            # number of folds in K-fold
                  prediction = FALSE,    # return the prediction using the final model 
                  showsd = TRUE,        # standard deviation of loss across folds
                  stratified = TRUE,    # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print.every.n = 50, 
                  early.stop.round = 10
)

xgb_1 = xgboost(data = x.matrix,
                label = y.matrix,
                params = xgb_params_1,
                nrounds = 1821,  # max number of trees to build
                verbose = TRUE,                                         
                print.every.n = 50,
                early.stop.round = NULL
)


preds <- predict(xgb_1, test.matrix)

dimnames <- list(c("predict_0", "predict_1", "predict_2"), NULL)
preds <- matrix(preds, nrow=num.classes, ncol=length(preds)/num.classes, dimnames = dimnames)
preds <- t(preds)
out <- cbind(id=test$id, as.data.frame(preds))
write.csv(out, file="submissions/submission-xgb.csv", row.names=FALSE, quote=FALSE)
