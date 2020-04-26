suppressWarnings(library(dummies))
suppressWarnings(library(xgboost))
suppressWarnings(library(methods))
suppressWarnings(library(reshape2))

dataPath <- "./"
train <- read.csv(paste(dataPath,"train_sample.csv",sep="/"),header=T)
test <- read.csv(paste(dataPath,"test_sample.csv",sep="/"),header=T)

require(xgboost)
require(methods)
require(reshape2)

#read data to the R console

featureclass = rep('numeric',93)
colclasstrain = c('integer',featureclass,'character')
colclasstest = c('integer',featureclass)

#keep record of the test id for final output

id = test[,1]  

#remove the id column

test = test[,-1]

#convert the target from character into integer starting from 0 

target = train$target
classnames = unique(target)

#remove the target the from train

train = train[,-ncol(train)]

#convert dataset into numeric Matrix format

trainMatrix <- data.matrix(train)
testMatrix <- data.matrix(test)
trainMatrix<-scale(trainMatrix)
testMatrix<-scale(testMatrix)


#cross-validation to choose the parameters

numberOfClasses <- 9

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)

#cv.nround <- 200
#cv.nfold <- 10
#bst.cv = xgb.cv(param=param, data = trainMatrix, label = target, 
#                nfold = cv.nfold, nrounds = cv.nround)

#nround <- which(bst.cv$test.mlogloss.mean==min(bst.cv$test.mlogloss.mean))

#train the model
nround = 195            #this number is the number of trees when test mlogloss is minimum during cross-validation
bst = xgboost(data = trainMatrix, label = target, param=param, nrounds = nround)

#predict the model

ypred = predict(bst, testMatrix)


#prepare for output

predMatrix <- data.frame(matrix(ypred, ncol=9, byrow=TRUE))
colnames(predMatrix) = classnames
res<-data.frame(id, predMatrix)
write.csv(res, 'result_v2.csv', quote = F, row.names = F)

