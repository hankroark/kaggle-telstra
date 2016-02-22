library(h2o)
library(readr)
h2o.init(ip="127.0.0.1", port=54323)

h2o.train <- h2o.uploadFile("data-postprep/train.csv")
h2o.test  <- h2o.uploadFile("data-postprep/test.csv")

h2o.train$fault_severity <- as.factor(h2o.train$fault_severity)
y <- "fault_severity"
x <- setdiff(names(h2o.train), y)  

gbm.model <- h2o.gbm(x=x, y=y, training_frame = h2o.train, distribution = "multinomial") # leader board reports 0.63500

# make predictions
predictions <- h2o.predict(gbm.model, h2o.test)

# added id needed for submission
predictions$id <- h2o.test$id

# prep data for writing out
df.preds <- as.data.frame(predictions)
df.preds$predict <- NULL
names(df.preds) <- c("predict_0","predict_1","predict_2","id")
write.csv(df.preds[c(4,1,2,3)], file="submissions/submission-h2o-gbm-22Feb.csv", row.names=FALSE, quote=FALSE)
