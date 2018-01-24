#libraries needed

library(caret)
library(rpart)
library(RColorBrewer)
library(rattle)
library(rpart.plot)
library(randomForest)
library(knitr)
library(e1071)
library(gbm)
library(plyr)

#Making data folder, downloading and loading the data

if (!file.exists("PMLdata")) {dir.create("PMLdata")}
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
trainfile <- "./PMLdata/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testfile <- "./PMLdata/pml-testing.csv"
trainfile <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testfile <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))



#Cleaning the Data

features <- names(testfile[,colSums(is.na(dt_testing)) == 0])[8:59]

# Only use features used in testing cases.
trainfile <- trainfile [,c(features,"classe")]
testfile <- testfile[,c(features,"problem_id")]

dim(trainfile); dim(testfile);

#Partitioning the Dataset

set.seed(11111)

Train01 <- createDataPartition(trainfile$classe, p=0.75, list=FALSE)
Train_A <- trainfile[Train01,]
Test_A <- trainfile[-Train01,]

dim(Train_A); dim(Test_A);

#Building the Decision Tree Model

set.seed(11111)
TreeMod <- rpart(classe ~ ., data = Train_A, method="class", control = rpart.control(method = "cv", number = 5))
fancyRpartPlot(TreeMod)

#Predicting with the Decision Tree Model

set.seed(11111)

predict_A <- predict(TreeMod, Test_A, type = "class")
confusionMatrix(predict_A, Test_A$classe)







#Building the Random Forest Model

set.seed(11111)

modFitRF <- randomForest(classe ~ ., data = Train_A, method = "rf", importance = T, trControl = trainControl(method = "cv", classProbs=TRUE,savePredictions=TRUE,allowParallel=TRUE, number = 10))

plot(modFitRF)


#Predicting with the Random Forest Model

prediction <- predict(modFitRF, Test_A, type = "class")
confusionMatrix(prediction, Test_A$classe)


#Predicting with the Testing Data (pml-testing.csv)

#Random Forest Prediction

predictionRF <- predict(modFitRF, testfile)
predictionRF
