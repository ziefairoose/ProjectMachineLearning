Summary
-------

Use this link to view html output
<http://htmlpreview.github.io/?https://github.com/ziefairoose/ProjectMachineLearning/blob/master/machinelearningproject.html>

The conceptual goal of this assignment is to do exploratory data
analysis and preprocess data before performing trainings and
predictions. The end goal is to accurately predict the type of exercises
based on the movement data. Several prediction models were used and the
model that yielded the best result were chosen.

The secondary goal of this assignment is to walk through the technical
steps that were presented in the Machine Learning lectures. Many steps
are explained at the basic levels to serve as notes for future reviews.
Due to bugs, the models were executed manually/individually and the
random forest yielded the best accuracy. Therefore, the output of the
script only contain the result for the random forest model.

The out of sample error rate is expected to be higher than the training
model. OR The prediction accuracy is expected to be less for out of
sample data.

### Bugs

There are unexplained bugs in the caret package, this script cannot run
all the models unattended (or sometimes individual model). Sometimes,
the models would run, sometimes they error out, sometimes the models
would be completed but the script would not return to the command line
and execute the next command. Often, when the training step hanged for 5
minutes, pressing the stop button will cause it to return to the command
line, and then can manually proceed to the next step.

Instructions
------------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it.

In this project, the goal will be to use data from accelerometers on the
belt, forearm, arm, and dumbell of 6 participants. They were asked to
perform barbell lifts correctly and incorrectly in 5 different ways.
More information is available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

### Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source:
<http://groupware.les.inf.puc-rio.br/har>. If you use the document you
create for this class for any purpose please cite them as they have been
very generous in allowing their data to be used for this kind of
assignment.

The goal of the project is to predict the manner in which they did the
exercise. This is the "classe" variable in the training set. You may use
any of the other variables to predict with. You should create a report
describing how you built your model, how you used cross validation, what
you think the expected out of sample error is, and why you made the
choices you did. You will also use your prediction model to predict 20
different test cases.

1.  Your submission should consist of a link to a Github repo with your
    R markdown and compiled HTML file describing your analysis. Please
    constrain the text of the writeup to \< 2000 words and the number of
    figures to be less than 5. It will make it easier for the graders if
    you submit a repo with a gh-pages branch so the HTML page can be
    viewed online (and you always want to make it easy on graders :-).

2.  You should also apply your machine learning algorithm to the 20 test
    cases available in the test data above. Please submit your
    predictions in appropriate format to the programming assignment for
    automated grading. See the programming assignment for additional
    details.

### Reproducibility

Due to security concerns with the exchange of R code, your code will not
be run during the evaluation by your classmates. Please be sure that if
they download the repo, they will be able to view the compiled HTML
version of your analysis.

    #install.packages("doParallel")
    #library(doParallel)
    #cl <- makeCluster(detectCores())
    #registerDoParallel(cl)
    #rm(list=ls())

    library(caret)

    ## Warning: package 'caret' was built under R version 3.2.3

    ## Loading required package: lattice

    ## Warning: package 'lattice' was built under R version 3.2.2

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.2.2

    library(rpart)

    ## Warning: package 'rpart' was built under R version 3.2.2

    library(rpart.plot)

    ## Warning: package 'rpart.plot' was built under R version 3.2.3

    library(randomForest) #needed just in case caret's train function does not work properly

    ## Warning: package 'randomForest' was built under R version 3.2.3

    ## randomForest 4.6-12
    ## Type rfNews() to see new features/changes/bug fixes.

    # library(plyr)
    # library(gbm)
    # library(survival)
    # library(splines)
    # library(parallel)
    set.seed(1111)

### Load the data

    if(file.exists("training.csv")) {
        message("data files are present on hard drive")
      }  else {
        message("no data file found on hard drive, downloading from sources")
        trainURL <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        download.file(trainURL, destfile="training.csv")
        download.file(testURL, destfile="testing.csv")
      }

    ## data files are present on hard drive

    #trainData <- read.csv("training.csv")
    #str(trainData) #19622 obs. of  160 variables
    #many data points are empty or NA or (#DIV/0!), convert them to NA values
    trainData <- read.csv("training.csv", na.strings = c("NA", "", "#DIV/0!"))
    testData <- read.csv("testing.csv", na.strings = c("NA", "", "#DIV/0!"))

Pre-process Data
----------------

There seem to be many NA values from missing data, find out which
variables have too many missing values to be useful and throw them out.

When running the summary(trainData), the result table shows many
variables have 19xxx NA values, which means they are mostly junk
variables. In fact, every column that has missing values has more than
19000 missing values. Remove all columns with NAs. Remove the first 7
columns because they are not relevant data. If necessary, run
nearZeroVar on the data to determine if there are other unuseful
columns.

    #summary(trainData) 

    #loop through variables and remove junks
    naVector <-vector()
    for(i in 1:length(trainData)) { 
      if(any(is.na(trainData[, i]))) {
        naVector<-c(naVector,i)
      }
    }
    trainData<-trainData[,-naVector]
    trainData<-trainData[,-c(1:7)] #remove user ids and timestamps
    #head(trainData)
    #str(trainData)
    #nearZeroVar(trainData, saveMetrics = TRUE) #video lecture "covariate creation" at 11min mark. No near zero var was found, so no need to eliminate additional variables.

### Need to preprocess with PCA?

Below is the code for determining variable with high correlations. Many
correlated variables, will need PCA preprocess?

Other students reported on the forum that there is no benefit to include
PCA for random forest. From the actual results, excluding PCA increased
accuracy from 90% to 97%. So no need to include pca during data traing
after all.

    M<-abs(cor(trainData[,-53]))
    diag(M)<-0
    which(M>.8,arr.ind=T) #

### Partition Data

From community tips, split training Data into 3 sets (60/20/20). If use
a slow computer, lower training set to 15% to process in reasonable
time.

UPDATE: use the native randomForest function to train instead of the
train function in caret. This will be much faster and no crashes.

The course's TAs clarifications -\> use the downloaded training data to
train and test/cross validate. Use the testing set (downloaded) to
submit answer to the online tests.

    inTrain <- createDataPartition(trainData$classe, p = 0.6)[[1]]
    #inTrain <- createDataPartition(trainData$classe, p = 0.6, list=FALSE)#the "list"=F param is to turn off the result output as list, so no need to attach [[1]] at the end. 
    CV <- trainData[-inTrain,]
    training <- trainData[ inTrain,]
    inTrain <- createDataPartition(CV$classe, p = 0.5)[[1]]
    CVTest <- CV[ -inTrain,]
    CV <- CV[inTrain,]

    dim(training);dim(CVTest);dim(CV)

    ## [1] 11776    53

    ## [1] 3923   53

    ## [1] 3923   53

### The Random Forest model

Using the caret package to train takes longer and often experience
crashes or hang. The accuracy here is only 97%. The bigger training data
set can increase the accuracy (use more powerful computer to find out).

UPDATE: The native randomForest function can train faster without
crashes, and able to train a large data set in reasonable time.

    #rfFit <- train(classe~., data=training, method = "rf", prox=T, trControl = trainControl(method = "cv", number=3)) #trControl = trainControl(method = "cv", number=3)

    rfFit <- randomForest(classe ~. , data=training)


    #rfFit$finalModel
    rfPredict <- predict(rfFit, CV)
    confusionMatrix(rfPredict, CV$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    2    0    0    0
    ##          B    0  754    5    0    0
    ##          C    0    3  678   13    0
    ##          D    0    0    1  630    0
    ##          E    0    0    0    0  721
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9939          
    ##                  95% CI : (0.9909, 0.9961)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9923          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9934   0.9912   0.9798   1.0000
    ## Specificity            0.9993   0.9984   0.9951   0.9997   1.0000
    ## Pos Pred Value         0.9982   0.9934   0.9769   0.9984   1.0000
    ## Neg Pred Value         1.0000   0.9984   0.9981   0.9961   1.0000
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1922   0.1728   0.1606   0.1838
    ## Detection Prevalence   0.2850   0.1935   0.1769   0.1608   0.1838
    ## Balanced Accuracy      0.9996   0.9959   0.9931   0.9897   1.0000

    rfPredict1 <- predict(rfFit, CVTest)
    confusionMatrix(rfPredict1, CVTest$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1115    5    0    0    0
    ##          B    0  751    5    0    0
    ##          C    0    3  675   15    0
    ##          D    0    0    4  627    2
    ##          E    1    0    0    1  719
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9908          
    ##                  95% CI : (0.9873, 0.9936)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9884          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9991   0.9895   0.9868   0.9751   0.9972
    ## Specificity            0.9982   0.9984   0.9944   0.9982   0.9994
    ## Pos Pred Value         0.9955   0.9934   0.9740   0.9905   0.9972
    ## Neg Pred Value         0.9996   0.9975   0.9972   0.9951   0.9994
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2842   0.1914   0.1721   0.1598   0.1833
    ## Detection Prevalence   0.2855   0.1927   0.1767   0.1614   0.1838
    ## Balanced Accuracy      0.9987   0.9939   0.9906   0.9866   0.9983

The code below are for the tree partition model and the boost model,
they are not ran because the script crashes with too many models. The
tree model is only 55% accurate, and the boost model is 95% accurate
(compare to 97% accuracy rate of the RandomForest model)

    treeFit <- train(classe~., method="rpart",data=training) # preProcess=c("center","scale","pca"), trControl = trainControl(method = "cv", number=3)
    treeFit$finalModel
    treePredict <- predict(treeFit, CV)
    confusionMatrix(treePredict, CV$classe)
    treePredict1 <- predict(treeFit, CVTest)
    confusionMatrix(treePredict1, CVTest$classe)

    #better training function for tree, run faster and no error, must include "class" param to generate correct data for confusion matrix
    treeFit <- rpart (classe ~ ., data=training, method="class")
    treePredict <- predict(treeFit, CV, type="class")
    confusionMatrix(treePredict, CV$classe)


    boostFit <- train(factor(classe)~., method="gbm",data=training, verbose=F, trControl = trainControl(method = "cv", number=3)) # preProcess=c("center","scale","pca"), trControl = trainControl(method = "cv", number=3)
    boostFit$finalModel
    boostPredict <- predict(boostFit, CV)
    confusionMatrix(boostPredict, CV$classe)
    boostPredict1 <- predict(boostFit, CVTest)
    confusionMatrix(boostPredict1, CVTest$classe)

### Code for submission process

Process testData set the same way as the trainData set. Predict the
testData against the rfFit model, the outcome will be a dataframe of 20
answers. Use the outcome as the parameter of the project's given
function to create 20 individual text files. Use the text files to
submit.

    naVector <-vector()
    for(i in 1:length(testData)) { 
      if(any(is.na(testData[, i]))) {
        naVector<-c(naVector,i)
      }
    }
    testData<-testData[,-naVector]
    testData<-testData[,-c(1:7)] 
    str(testData)

    rfPredictTestData <- predict(rfFit, testData)
    rfPredictTestData

    pml_write_files = function(x){
      n = length(x)
      for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
      }
    }

    pml_write_files(rfPredictTestData)

### Conclusions

The random forest model provides the best prediction accuracy for this
data set. PCA preprocess actually decrease accuracy for this data set.
The accuracy for out of sample data is less than the accuracy rate of
the training sample (for all models, as expected). How much less is
depending upon the accuracy of the training set, which depends on the
size of the training data set.

### Notes/Tips

There are bugs in the caret package that prevent training to complete.
Use a smaller set of data to train if using a slower computer. If
training hangs for a long period, sometimes, pressing the stop button to
return to command line, the train object could be completed.
