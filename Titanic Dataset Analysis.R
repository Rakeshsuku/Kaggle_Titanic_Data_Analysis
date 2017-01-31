################################################################################
############### Titanic Dataset - Machine Learning from Disaster ###############
################################################################################
## Link: https://www.kaggle.com/c/titanic/data

setwd("/media/rakesh/G/Data Science/Practice Datasets/Titanic Dataset")
getwd()

## Training set has 2 observations with missing value (""). We set these to NAs.
titanicTrain <- read.csv("train.csv", na.strings=c("","NA")) 
titanicTest <- read.csv("test.csv", na.strings=c("","NA"))
## For Online code:
## titanicTrain <- read.csv("../input/train.csv", na.strings=c("","NA"))
## titanicTest <- read.csv("../input/test.csv",  na.strings=c("","NA"))

sum(is.na(titanicTrain$Embarked))
names(titanicTrain)
str(titanicTrain)

## Convert the variables Survived and Passenger Class to factors:
titanicTrain$Survived <- factor(titanicTrain$Survived, 
                                levels = c(0, 1), 
                                labels = c("Not_Survived", "Survived"))

titanicTrain$Pclass <- factor(titanicTrain$Pclass, 
                              levels = c(1, 2, 3), 
                              labels = c("1st_Class", "2nd_Class", "3rd_Class"))

str(titanicTrain)

##Convert Passenger Class to factors in Test data set.
titanicTest$Pclass <- factor(titanicTest$Pclass, 
                              levels = c(1, 2, 3), 
                              labels = c("1st_Class", "2nd_Class", "3rd_Class"))

str(titanicTest)
################################################################################
############################# Data Visualization: #############################

par(mfrow = c(1,2))
boxplot(titanicTrain$Age~titanicTrain$Survived, main = "Age", ylab = "Age")
boxplot(titanicTrain$Fare~titanicTrain$Survived, main = "Fare", ylab = "Fare")

## The above plots indicate that the age profile of passengers who survived the 
## titanic disaster is lower that that of those who did not. Also, passengers
## who survived the disaster paid a high amount of fare in general than those
## who did not survive the disaster.

plot(titanicTrain$Survived ~ titanicTrain$Sex, 
     main = "Survival~Gender", 
     xlab = "Sex",
     ylab = "Survival")

plot(titanicTrain$Survived ~ titanicTrain$Pclass, 
     main = "Survival~Passenger Class",
     xlab = "Passenger Class",
     ylab = "Survival")

## These plots indicates that women and high class passengers had better chance
## of survival in the titanic disaster.

## Lets examine the survival rates by gender varry across passenger classes?
library(ggplot2)
## Distribution of Men and Women in each class:
p1 <- ggplot(data = titanicTrain, mapping = aes(x = Sex))

## Distribution of Men and Women in each class:
p1 + geom_bar(aes(fill = Sex)) + 
    scale_fill_grey() + 
    facet_wrap(~Pclass) + 
    geom_text(stat='count',aes(label=..count..),vjust=-0.25)


p1 + geom_bar(position = "fill", aes(fill = Survived)) +
    facet_wrap(~Pclass)
## More than 90% of the women in 1st Class and 2nd Class survived the disaster.
## The percentage of women survived drops to about 50% in the third class.
## Almost 60% of the men in first class did not survive the disaster while about
## 80% of those in 2nd class and 3rd class also did not survive.


## Now let's examine how the survival rate varries across age group among males
## and females:
p2 <- ggplot(data = titanicTrain, mapping = aes(x = Age))

## Survival across age group
p2 + geom_histogram(aes(fill = Survived)) + 
    labs(title = "Survival rate across age group")

## Survival across age group by Sex and Passenger class:
p2 + geom_histogram(aes(fill = Survived)) + facet_wrap(Sex~Pclass) + 
    labs(title = "Survival rate across age group by Gender & Passenger class")


## Tabulate the distribution of passengers with Parents/Sibilings on-board
table (titanicTrain$Parch)
table (titanicTrain$SibSp)

## Next, let's examine how presence of family member onboard affects survival:
par(mfrow = c(1,1))
plot(titanicTrain$Survived ~ as.factor(titanicTrain$SibSp), 
     main = "Survival~Siblings Onboard",
     xlab = "No. of Siblings",
     ylab = "Survival")

plot(titanicTrain$Survived ~ as.factor(titanicTrain$Parch), 
     main = "Survival~Parents/Children Onboard",
     xlab = "No. of Parents/Children",
     ylab = "Survival")

plot(titanicTrain$Survived ~ as.factor(titanicTrain$Parch + titanicTrain$SibSp), 
     main = "Survival~Family Onboard",
     xlab = "No. of Family",
     ylab = "Survival")

p3 <- ggplot(data = titanicTrain, mapping = aes(x = as.factor(Parch + SibSp)))
p3 + geom_bar(position = "fill", aes(fill = Survived)) + facet_wrap(~Pclass) +
    labs(title = "Survival rate by No. of family members onboard across Class",
         x = "No. of family member onboard", y = "Percentage")

## Survival rates goes down for family members onboard >= 3.

################################################################################
########################## Model Tuning & Validation: ##########################

library(caret)
## Data Splitting:
set.seed(1)
trainingIndex <- createDataPartition(titanicTrain$Survived, p=0.75, 
                                     list = FALSE, times = 1)

trainingSet <- titanicTrain[trainingIndex, c("PassengerId","Survived", "Pclass", 
                                             "Sex", "Age", "SibSp", "Parch", 
                                             "Fare", "Cabin", "Embarked")]

validationSet <- titanicTrain[-trainingIndex, c("PassengerId", "Survived", 
                                                "Pclass", "Sex", "Age", "SibSp", 
                                                "Parch", "Fare", "Cabin", 
                                                "Embarked")]

testSet <- titanicTest[, c("PassengerId", "Pclass", "Sex", "Age", "SibSp", 
                           "Parch", "Fare", "Cabin", "Embarked")]
## PassengerId is required for cross checking results.

## Five repeats of 10-fold CV is used for tuning parameter estimation:
## See the help for trainControl() function at: 
## http://topepo.github.io/caret/model-training-and-tuning.html#custom

trainCtrl <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 5,
                          savePredictions = TRUE,
                          classProbs = TRUE)


#### Boosting ####

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7),
                       n.trees = seq(from = 150, to = 500, by = 50),
                       shrinkage = c(0.1, 0.01),
                       n.minobsinnode = 20)

set.seed(100)
gbmFit <- train(x = trainingSet[, -c(1, 2)],
                y = trainingSet$Survived,
                method = "gbm",
                tuneGrid = gbmGrid,
                verbose = FALSE,
                trControl = trainCtrl)

gbmFit
gbmFit$finalModel
validationSet$gbmPred <- predict(gbmFit, newdata = validationSet[, -c(1, 2)])
table(validationSet$gbmPred, validationSet$Survived)
## Validation Set Accuracy:
mean(validationSet$gbmPred == validationSet$Survived) ## 77.02%
## Accuracy among Survived & Not_Survived population:
with(validationSet[validationSet$Survived == "Survived", ], 
     mean(gbmPred == Survived))
## 57.64% Accuracy in prediction of Survived population

with(validationSet[validationSet$Survived == "Not_Survived", ], 
     mean(gbmPred == Survived))
## 89.51% Acuracy in prediction on Not_Survived population.

## Test set Prediction for GBM Model:
testSet$gbmPred <- predict(gbmFit, newdata = testSet[, -1])
table(testSet$gbmPred) 
## 128 Survived and 290 did not Survive as per GBM prediction 
## (Validation Set Accuracy: 77.02%).

#### Random Forest Model ####

library(randomForest)
rfGrid <- expand.grid(mtry = c(2:7)) 

## train() can only tune over mtry values for method = "rf" (randomForest).
## randomForest package cannot handle missing values. Age has 43 NA values.
## hence medianImpute is used to impute missing values.
## Also, randomForest cannot handle categorical variables with more than 53 
## levels. Hence removed "Cabin" (col 9)predictor from training and test set.

sapply(validationSet, FUN = function(x) {sum(is.na(x))})
## There are 43 missing values for Age variable, we impute the missing values 
## to apply random forest.

colNames <- names(trainingSet[, -c(1, 2, 9)]) ##Predictors for rf model
set.seed(100)

sapply(trainingSet[, colNames], FUN = function(x) {sum(is.na(x))})

## We remove the 2 sample with NA for Embarked variable. NA values in Age are
## imputed using median impute.
rfFit <- train(x = trainingSet[!is.na(trainingSet$Embarked), colNames],
               y = trainingSet[!is.na(trainingSet$Embarked), ]$Survived,
               method = "rf",
               tuneGrid = rfGrid,
               preProcess = "medianImpute",
               verbose = FALSE,
               ntree =2000,
               trControl = trainCtrl,
               na.action = na.omit)
rfFit
rfFit$finalModel
validationSet$rfPred <- predict(rfFit, newdata = validationSet[, colNames])
table(validationSet$rfPred, validationSet$Survived)
## Accuracy:
mean(validationSet$rfPred == validationSet$Survived) ## 81.08%

## Accuracy among Survived & Not_Survived population:
with(validationSet[validationSet$Survived == "Survived", ], 
     mean(rfPred == Survived))
## 71.76% Accuracy in prediction of Survived population

with(validationSet[validationSet$Survived == "Not_Survived", ], 
     mean(rfPred == Survived))
## 86.86% Acuracy in prediction on Not_Survived population.

## Test set Prediction for GBM Model:
testSet$rfPred <- predict(rfFit, newdata = testSet[, colNames])

table(testSet$rfPred) 
## 134 Survived and 284 did not Survive as per rf prediction 
## (Validation Set Accuracy: 81.08%).

submission <- data.frame(PassengerId = testSet$PassengerId,
                         Survived = unclass(testSet$rfPred)-1)

head(submission)

write.csv(submission, file = "random_forest_r_submission.csv", 
          row.names=FALSE)

