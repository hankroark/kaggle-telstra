library(h2o)
library(readr)
h2o.init()

h2o.train <- h2o.uploadFile("data-postprep/train.csv")
h2o.test  <- h2o.uploadFile("data-postprep/test.csv")

h2o.train$fault_severity <- as.factor(h2o.train$fault_severity)
y <- "fault_severity"
x <- setdiff(names(h2o.train), y)  

gbm.model <- h2o.gbm(x=x, y=y,
                    training_frame = h2o.train, nfolds = 5)
summary(gbm.model)

# make predictions
predictions <- h2o.predict(gbm.model, h2o.test)

# added id needed for submission
predictions$id <- h2o.test$id

# prep data for writing out
df.preds <- as.data.frame(predictions)
df.preds$predict <- NULL
names(df.preds) <- c("predict_0","predict_1","predict_2","id")
write.csv(df.preds[c(4,1,2,3)], file="submissions/submission-h2o-gbm-6Feb.csv", row.names=FALSE, quote=FALSE)




# write_csv(train.removedconstantcolumns, "data/train_prepared.csv")
# write_csv(test.augmented, "data/test.augmented")
# 
# multilogloss.fun <- function(true, predicted) {
#   MultiLogLoss(y_true=true, y_pred=attr(predicted, "probabilities"))
# }
# 
# # model using svm
# svm.model <- svm(fault_severity ~ . - id - location, data = train.removedconstantcolumns, scale=TRUE, probability=TRUE)
# pred <- predict(svm.model, train.removedconstantcolumns, probability=TRUE)
# multilogloss.fun(train.removedconstantcolumns$fault_severity,pred)  # 1.238027, no CV, lb = 0.76910
# 
# #summary(svm.model)
# #table(predict=pred, truth=train.removedconstantcolumns$fault_severity)
# 
# preds <- predict(svm.model, test.augmented, probability = TRUE)
# preds.df <- as.data.frame(attr(preds, "probabilities"))
# preds.df$id <- test.augmented$id
# names(preds.df) <- c("predict_1","predict_0","predict_2","id")
# write_csv(preds.df[c(4,2,1,3)], path="submissions/submission7.csv")
# 
# 
# # hyperparameter search
# obj <- tune.svm(fault_severity ~ . - id - location, data = train.removedconstantcolumns, scale=TRUE, probability=TRUE, cost = sqrt(10)^(0:6), tune.control=tune.control(error.fun=multilogloss.fun))
# summary(obj)
# plot(obj)
# 
# pred <- predict(obj$best.model, train.removedconstantcolumns, probability=TRUE)
# multilogloss.fun(train.removedconstantcolumns$fault_severity,pred)  # 1.358843
# 
# 
# pred.test <- predict(obj$best.model, test.augmented, probability=TRUE)  # C=100, gamma = 0.002531646, radial, error = 0.2624304, lb = 0.76910
# preds.df <- as.data.frame(attr(preds.test, "probabilities"))
# preds.df$id <- test.augmented$id
# names(preds.df) <- c("predict_1","predict_0","predict_2","id")
# write_csv(preds.df[c(4,2,1,3)], path="submissions/submission8.csv")
# 
