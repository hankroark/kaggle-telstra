setwd("~/Documents/github/telstra/scripts")

library(data.table)
library(Matrix)
library(glmnet)
library(doMC)
library(h2o)
library(cvTools)
library(pracma)
library(ggplot2)


sample.submission <- fread("../data/sample_submission.csv")
# > head(sample.submission)
# id predict_0 predict_1 predict_2
# 1: 11066         0         1         0
# 2: 18000         0         1         0
# 3: 16964         0         1         0
# 4:  4795         0         1         0
# 5:  3392         0         1         0
# 6:  3795         0         1         0


train <- fread("../data/train.csv")
# > head(train)
# id     location fault_severity
# 1: 14121 location 118              1
# 2:  9320  location 91              0
# 3: 14394 location 152              1
# 4:  8218 location 931              1
# 5: 14804 location 120              0
# 6:  1080 location 664              0

test  <- fread("../data/test.csv")
# > head(test)
# id     location
# 1: 11066 location 481
# 2: 18000 location 962
# 3: 16964 location 491
# 4:  4795 location 532
# 5:  3392 location 600
# 6:  3795 location 794

event.type <- fread("../data/event_type.csv")
# > head(event.type)
# id    event_type
# 1: 6597 event_type 11
# 2: 8011 event_type 15
# 3: 2597 event_type 15
# 4: 5022 event_type 15
# 5: 5022 event_type 11
# 6: 6852 event_type 11

log.feature <- fread("../data/log_feature.csv")
# id log_feature volume
# 1: 6597  feature_68      6
# 2: 8011  feature_68      7
# 3: 2597  feature_68      1
# 4: 5022 feature_172      2
# 5: 5022  feature_56      1
# 6: 5022 feature_193      4

resource.type <- fread("../data/resource_type.csv")
# > head(resource.type)
# id   resource_type
# 1: 6597 resource_type 8
# 2: 8011 resource_type 8
# 3: 2597 resource_type 8
# 4: 5022 resource_type 8
# 5: 6852 resource_type 8
# 6: 5611 resource_type 8

severity.type <- fread("../data/severity_type.csv")
# > head(severity.type)
# id   severity_type
# 1: 6597 severity_type 2
# 2: 8011 severity_type 2
# 3: 2597 severity_type 2
# 4: 5022 severity_type 1
# 5: 6852 severity_type 1
# 6: 5611 severity_type 2

# Simplest thing I can think of is to merge all the frames together by id, being sure to keep all the rows in train and test
expected.rows.train <- nrow(train)
expected.rows.test  <- nrow(test)

train[,':=' (predict_0 = ((fault_severity==0)*1), predict_1 = ((fault_severity==1)*1), predict_2 = ((fault_severity==2)*1))]

convert_frame <- function(frame) {
  
  event.type$event_type <- sub(" ", "_", event.type$event_type)
  event.type.cast <- dcast(event.type, id ~ event_type, fun.aggregate = function(x) sum(x != ""))
  frame.augmented <- merge(frame, event.type.cast, by="id", all.x=TRUE)
  
  log.feature$log_feature <- sub(" ", "_", log.feature$log_feature)
  log.feature.cast <- dcast(log.feature, id ~ log_feature, value.var = "volume", fill=0)
  frame.augmented <- merge(frame.augmented, log.feature.cast, by="id", all.x=TRUE)
  
  resource.type$resource_type <- sub(" ", "_", resource.type$resource_type)
  resource.type.cast <- dcast(resource.type, id ~ resource_type, fun.aggregate = function(x) sum(x != ""))
  frame.augmented <- merge(frame.augmented, resource.type.cast, by="id", all.x=TRUE)
  
  severity.type$severity_type <- sub(" ", "_", severity.type$severity_type)
  severity.type.cast <- dcast(severity.type, id ~ severity_type, fun.aggregate = function(x) sum(x != ""))
  frame.augmented <- merge(frame.augmented, severity.type.cast, by="id", all.x=TRUE)
  
  frame.augmented
}

train.augmented <- convert_frame(train)
test.augmented <- convert_frame(test)

x.columns <- tail(names(train.augmented),n=-6)
y.columns.glm <- c("predict_0", "predict_1", "predict_2")
y.column.h2o <- "fault_severity"

x.matrix <- Matrix( as.matrix( train.augmented[,x.columns,with=FALSE] ), sparse = TRUE)
y.matrix <- as.matrix( train.augmented[,y.columns.glm,with=FALSE])

fit <- glmnet(x=x.matrix, y=y.matrix, family="multinomial", alpha=1)
plot(fit, xvar="lambda", label=TRUE, type.coef="coef")

registerDoMC(cores=8)
cvfit <- cv.glmnet(x=x.matrix, y=y.matrix, parallel=TRUE, family="multinomial", alpha=1, lambda.min.ratio=1e-8, nlambda=20, maxit=100000000)
plot(cvfit)

#### H2O 
h2o.init(nthreads=-1)

h2o.train <- as.h2o(train.augmented)
h2o.train$fault_severity <- as.factor(h2o.train$fault_severity)
h2o.test <- as.h2o(test.augmented)

k <- 10
folds <- cvFolds(NROW(train.augmented), K=k)

h2o.folds <- as.h2o(folds$which)
h2o.train <- h2o.cbind(h2o.train,h2o.folds)

fold.column <- "x"

###### Need for GLM ####
cv.train.frames <- c()
cv.valid.frames <- c()
for(fold in 1:k) {
  print(fold)
  cv.train.frames <- c(cv.train.frames, h2o.train[h2o.train$x != fold,])
  cv.valid.frames <- c(cv.valid.frames, h2o.train[h2o.train$x == fold,])
}

lambdas.to.search <- logspace(-5, -1, n=20)

lambdas <- c()
logloss <- c()
for( lvalue in lambdas.to.search ) {
  print(lvalue)
  for( fold in 1:k ) {
    print(fold)
    cvmodel <- h2o.glm(x=x.columns, y=y.column.h2o, training_frame = cv.train.frames[[fold]], family="multinomial", alpha = 1, lambda = lvalue)
    performance <- h2o.performance(cvmodel, data = cv.valid.frames[[fold]])
    ll <- h2o.logloss(performance)
    lambdas <- c(lambdas, lvalue)
    logloss <- c(logloss, ll)
  }
}

cvresults <- data.frame(lambda = as.factor(lambdas), loss = logloss)
ggplot(cvresults, aes(lambda, loss)) + geom_boxplot()
best_lambda <- lambdas.to.search[[15]]  # lb 0.69493, submission 1, lambda within IQR of mean of best lambda=0.008858668
best_lambda <- lambdas.to.search[[12]]  # lb 0.67242, submission 2, best mean loss lambda=0.002069138
best_lambda <- lambdas.to.search[[13]]  # lb 0.67234, submission 3
best_lambda <- lambdas.to.search[[14]]  # lb 0.68100, submission 4

fit <- h2o.glm(x=x.columns, y=y.column.h2o, training_frame=h2o.train, family="multinomial", alpha=1, lambda=best_lambda)
predictions <- h2o.predict(fit, h2o.test)
predictions$id <- h2o.test$id

df.preds <- as.data.frame(predictions)
df.preds$predict <- NULL
names(df.preds) <- c("predict_0","predict_1","predict_2","id")
write.csv(df.preds[c(4,1,2,3)], file="../submissions/submission4.csv", row.names=FALSE, quote=FALSE)
### END GLM

gbm.hyper.params <- list(ntrees=c(1000,300,100), learn_rate=c(0.01,0.03,0.1), max_depth=c(2,5,10), col_sample_rate_per_tree=c(0.7,1),  sample_rate=c(0.7,1))
gbm.hyper.params <- list(ntrees=c(1000,300,100), learn_rate=c(0.01,0.03,0.1), max_depth=c(2,5,10), sample_rate=c(0.7,1))

gbm.grid <- h2o.grid(algorithm = "gbm", grid="gbm.grid", x=x.columns, y=y.column.h2o, nfolds=10, training_frame=h2o.train, distribution="multinomial", hyper_params = gbm.hyper.params)

grid_models <- lapply(gbm.grid@model_ids, function(model_id) { model = h2o.getModel(model_id) })
for (i in 1:length(grid_models)) {
  print(i)
  print(sprintf("logloss: %f", h2o.logloss(grid_models[[i]], xval=TRUE)))
}
best.gbm <- grid_models[[17]]  #lb 0.55763
best.gbm

predictions <- h2o.predict(best.gbm, h2o.test)
predictions$id <- h2o.test$id

df.preds <- as.data.frame(predictions)
df.preds$predict <- NULL
names(df.preds) <- c("predict_0","predict_1","predict_2","id")
write.csv(df.preds[c(4,1,2,3)], file="../submissions/submission5.csv", row.names=FALSE, quote=FALSE)
