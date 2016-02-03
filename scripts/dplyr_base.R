library(dplyr)
library(tidyr)
library(readr)
library(e1071)
library(MLmetrics)

sample.submission <- read_csv("data/sample_submission.csv")
# head(sample.submission)
# Source: local data frame [6 x 4]
# 
# id predict_0 predict_1 predict_2
# (int)     (int)     (int)     (int)
# 1 11066         0         1         0
# 2 18000         0         1         0
# 3 16964         0         1         0
# 4  4795         0         1         0
# 5  3392         0         1         0
# 6  3795         0         1         0

train <- read_csv("data/train.csv")
# > head(train)
# Source: local data frame [6 x 3]
#
# id     location fault_severity
# (int)        (chr)          (int)
# 1 14121 location 118              1
# 2  9320  location 91              0
# 3 14394 location 152              1
# 4  8218 location 931              1
# 5 14804 location 120              0
# 6  1080 location 664              0

test  <- read_csv("data/test.csv")
# > head(test)
# Source: local data frame [6 x 2]
# 
# id     location
# (int)        (chr)
# 1 11066 location 481
# 2 18000 location 962
# 3 16964 location 491
# 4  4795 location 532
# 5  3392 location 600
# 6  3795 location 794

event.type <- read_csv("data/event_type.csv")
# > head(event.type)
# Source: local data frame [6 x 2]
# 
# id    event_type
# (int)         (chr)
# 1  6597 event_type 11
# 2  8011 event_type 15
# 3  2597 event_type 15
# 4  5022 event_type 15
# 5  5022 event_type 11
# 6  6852 event_type 11

log.feature <- read_csv("data/log_feature.csv")
# > head(log.feature)
# Source: local data frame [6 x 3]
# 
# id log_feature volume
# (int)       (chr)  (int)
# 1  6597  feature 68      6
# 2  8011  feature 68      7
# 3  2597  feature 68      1
# 4  5022 feature 172      2
# 5  5022  feature 56      1
# 6  5022 feature 193      4

resource.type <- read_csv("data/resource_type.csv")
# > head(resource.type)
# Source: local data frame [6 x 2]
# 
# id   resource_type
# (int)           (chr)
# 1  6597 resource_type 8
# 2  8011 resource_type 8
# 3  2597 resource_type 8
# 4  5022 resource_type 8
# 5  6852 resource_type 8
# 6  5611 resource_type 8

severity.type <- read_csv("data/severity_type.csv")
# > head(severity.type)
# Source: local data frame [6 x 2]
# 
# id   severity_type
# (int)           (chr)
# 1  6597 severity_type 2
# 2  8011 severity_type 2
# 3  2597 severity_type 2
# 4  5022 severity_type 1
# 5  6852 severity_type 1
# 6  5611 severity_type 2

# Simplest thing I can think of is to merge all the frames together by id, being sure to keep all the rows in train and test
expected.rows.train <- nrow(train)
expected.rows.test  <- nrow(test)

# fix up the spaces, since event_type will become columns names
event.type$event_type <- sub(" ", "_", event.type$event_type)
# add a dummy count to event.type for each occurance, so spread will work properly
event.type$count <- 1
# need to spread since id's appear more than once
# > nrow(event.type)
# [1] 31170
# > length(unique(event.type$id))
# [1] 18552
event.type.spread <- spread(event.type, event_type, count, fill = 0)  

# fix up the spaces, since log_feature will become columns names
log.feature$log_feature <- sub(" ", "_", log.feature$log_feature)
# need to spread since id's appear more than once
log.feature.spread <- spread(log.feature, log_feature, volume, fill=0) 

# and so on
resource.type$resource_type <- sub(" ", "_", resource.type$resource_type)
resource.type$count <- 1
resource.type.spread <- spread(resource.type, resource_type, count, fill=0)

# and so on
severity.type$severity_type <- sub(" ", "_", severity.type$severity_type)
# don't have to spread severity type since there are same number of rows as there are unique ids
# but i will do it anyway to one encode the data for modeling
severity.type$count <- 1
severity.type.spread <- spread(severity.type, severity_type, count, fill=0)

#########
# I didn't blow out location because their are locations in test that aren't in training, so can't generalize
# and therefore won't use in location in modeling
#########
if( length(setdiff(test$location, train$location)) > 0 ) { warning("there are locations in test not observed in train, be careful using location for modeling")}

convert_frame <- function(frame) {
  temp <- left_join(frame, event.type.spread, by="id")
  temp <- left_join(temp, log.feature.spread, by="id")
  temp <- left_join(temp, resource.type.spread, by="id")
  temp <- left_join(temp, severity.type.spread, by="id")
  temp
}

train.augmented <- convert_frame(train)
train.augmented$fault_severity <- as.factor(train.augmented$fault_severity)
test.augmented <- convert_frame(test)

if( expected.rows.train != nrow(train.augmented) | expected.rows.test != nrow(test.augmented) ) { warning("train and/or test length do not match before merge") }

# remove constant columns
columnstokeep <- apply(train.augmented, 2, var, na.rm=TRUE) != 0
columnstokeep["location"] = TRUE
train.removedconstantcolumns <- train.augmented[,columnstokeep]

# End data prep
# Start modeling

library(h2o)
h2o.init()

h2o.train <- as.h2o(train.removedconstantcolumns)
h2o.test  <- as.h2o(test.augmented)
h2o.train$fault_severity <- as.factor(h2o.train$fault_severity)

rf.model <- h2o.randomForest(x=4:length(h2o.train), y=3, max_depth = 40,
                             training_frame = h2o.train, nfolds = 5)
summary(rf.model)

# make predictions
predictions <- h2o.predict(rf.model, h2o.test)

# added id needed for submission
predictions$id <- h2o.test$id

# prep data for writing out
df.preds <- as.data.frame(predictions)
df.preds$predict <- NULL
names(df.preds) <- c("predict_0","predict_1","predict_2","id")
write.csv(df.preds[c(4,1,2,3)], file="submissions/submission-rf.csv", row.names=FALSE, quote=FALSE)


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
