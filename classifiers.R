#Using the Wine data set and iris data set
#features : Task: Classification, Type : Multivariate, number of instances: 178

install.packages("corrplot",  dependencies	=	TRUE)
install.packages("tree",dependencies = T)
install.packages("rpart",dependencies = T)
install.packages("ggplot2", dependencies = T)
install.packages("e1071", dependencies = T)
install.packages("neuralnet", dependencies = T)
install.packages("klar",dependencies = T)
install.packages("caret",dependencies = T)
install.packages("MASS")

require(corrplot)
require(ggplot2)
require(reshape2)
##########For decision trees

#library(rattle) # GUI for building trees and fancy tree plot
require(tree)
library(rpart) # Popular decision tree algorithm
library(rpart.plot) # Enhanced tree plots
library(party) # Alternative decision tree algorithm
library(partykit) # Convert rpart object to BinaryTree
#library(RWeka) # Weka decision tree J48.
#library(C50) # Original C5.0 implementation.

#for Perceptron
require(neuralnet)
#for naive bayes
install.packages('e1071', dependencies = TRUE)
install.packages("klaR")
install.packages("caret")
library(class) 
library(e1071)
library(klaR)
library(caret)
library(MASS)

#for svm
library(e1071)
data("iris")
#for neural networks
library(neural)
library(neuralnet)
library(NeuralNetTools)
library(ROCR) #this is for getting the roc curve, to determine which should be value of choosing value of threshold

getwd()
setwd("C:/Users/vivek/Desktop/Fall 2016/Machine Learning/Assignment/Assignment 2")

#############################################Part 1 
##########################
#for picking up data 
wine_data<-read.csv(file.choose(), header = F) #freedom to pick the data set from any place
#creating copy of data
wine_data_copy<-wine_data

#to view the data
View(wine_data)
#to understand the dimensions of the data
dim(wine_data)
#to get an overall summary of the data
summary(wine_data)
#to get details of each column, type of data contained, overall count of attributes
str(wine_data)
#to get number of NA's in the data set
sum(is.na(wine_data))
#incase there are , say, na's then we have to neglect them by using complete.cases() function
#or a better and beautiful way to look at na's in each of the column 
#of the entire data frame 
na_count <-sapply(wine_data, function(y) sum(is.na(y)))
na_count <- data.frame(na_count)

#Pre Processing 
#giving proper names to the dataframe columns
names(wine_data)
colnames(wine_data)<-c("Class","Alcohol","Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols","Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline")

names(wine_data)


#############################################Part 2
##########################
#to be written in word
##########################


#############################################Part 3:
### 
#Some more pre processing and stuff like histograms and correlations
###
#number of classes n instances of each and every column
xtabs( ~ Class, data = wine_data)
#if you want it for a specific column, use table(dataframe$columnname)
#for scatterplot
plot(wine_data)

#for all correlation plots
All_in_one<-cor(wine_data)
corrplot(All_in_one, method="circle")
corrplot(All_in_one,method  =	"number")
#for all histograms, to get an idea of distributions
d <- melt(wine_data[,-c(1)])
ggplot(d,aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram()

#question How to scale data? As two variables have very varied scales as seen from the histograms



################################################Part 4### 

#############################################Training 

# 
#  Decision Trees
#  Naïve Bayes Classifiers

#  Perceptron (Single Linear Classifier)
#  Neural Net
#  Support Vector Machines




########################## Decision Trees #####################
#Decision trees using rpart
?rpart
#second copy of data , before partitioning

wine_data_copy2<-wine_data

#change class columns to levels or factors 
wine_data$Class<-as.factor(wine_data$Class)

mydtrpart1 = rpart(Class ~ .,data = wine_data) #this didnt work, my guess would be that the issue lies with the class column

#update 2 this is now working
#this is just to check whether rpart is working or not 
#to tune parameters shortly 

plotcp(mydtrpart1)#to understand how the relative error changes based on number of splits

printcp(mydtrpart1) # to understand in terms of numerical as opposed to graphical, these values can be used to understand pruning

mydtrpart2=prune(mydtrpart1,cp= 0.01)
# using training test 80-20 share, # replace this with iterative function.
plot(mydtrpart2)
text(mydtrpart2, pretty = 0)

train <- sample(1:nrow(wine_data), 0.80 * nrow(wine_data))

DTree1 <- rpart(Class ~ ., data = wine_data[train, ], method = 'class')
printcp(DTree1)
DTree1<-prune( DTree1, cp= 0.01)
plot(DTree1)
text(DTree1, pretty = 0)

#summary(DTree1) #uncomment in actual submit
#dput(summary(DTree1),file="summary_DT8020.txt",control="all") #parsing error
#saving detailed results to a text file
#out <- capture.output(summary(DTree1)) 
#cat("My title", out, file="summary_of_myDT.txt", sep="n", append=TRUE)
# test data performance 
DtreePred <- predict(DTree1, wine_data[-train, ], type = 'class')
t<-table(DtreePred, wine_data[-train, ]$Class)
accuracydt<-sum(diag(t))/sum(t)
t
accuracydt

#res<-as.data.frame(table(DtreePred, wine_data[-train, ]$Class))
#prop.table(table(DtreePred, wine_data[-train, ]$Class))#for fractions 
#t<-table(DtreePred, wine_data[-train, ]$Class)
#Accuracy calculation function to be written
# 
# loop over data frame, if dtree value == var 2 , store the freq value in sum, 
# accuracy = sum/(total sum of values in frequency)
# better alternate to calculate accuracy 

#store 5 test-training random sample values in vector or array, run loop, access 8020 8515 9010 and repeat 
#store results in dataframe
#further play with parameters method can be anova or poisson or class or exp 
##########################################################




######### Naive Beyes ######### 
##using klar and caret packages
?naiveBayes


nbsample<-sample(1:nrow(wine_data), 0.80 * nrow(wine_data))

nbtrain<- wine_data[nbsample, ]
nbtest<-wine_data[-nbsample,]

nbtrainclass<-nbtrain$Class
nbtestclass<-nbtest$Class

#model = train(nbtrain,nbtrainclass,'nb',trControl=trainControl(method='cv',number=10))
# 
# t<-(table(predict(model$finalModel,nbtrain)$class,nbtestclass))
# t
# print("accuracy")
# sum((diag(t)))/sum(t)*100 

##trial nb on wine data ,using train() this is giving 100% accuracy , cross c

spam<-wine_data
sub = sample(nrow(spam), floor(nrow(spam) * 0.80))
train = spam[sub,]
test = spam[-sub,]

xTrain = train
yTrain = train$Class

xTest = test 
yTest = test$Class

model = train(xTrain,yTrain,'nb',trControl=trainControl(method='cv',number=10))
#prop.table(table(predict(model$finalModel,xTest)$class,yTest))
tt<-table(predict(model$finalModel,xTest)$class,yTest)
tt
print("accuracy =")
sum((diag(tt)))/sum(tt)*100 
summary(model)

## trial nb on iris data for validation and peace of mind
data(iris)
# define an 80%/20% train/test split of the dataset
split=0.80
trainIndex <- createDataPartition(iris$Species, p=split, list=FALSE)
data_train <- iris[ trainIndex,]
data_test <- iris[-trainIndex,]
# train a naive bayes model
model <- NaiveBayes(Species~., data=data_train)
# make predictions
x_test <- data_test[,1:4]
y_test <- data_test[,5]
predictions <- predict(model, x_test)
# summarize results
confusionMatrix(predictions$class, y_test)

#trial nb on wine data with naivebayes 

# define an 80%/20% train/test split of the dataset, manually updating split values if needed
split=0.80
trainIndex <- createDataPartition(wine_data$Class, p=split, list=FALSE)
data_train <- wine_data[ trainIndex,]
data_test <- wine_data[-trainIndex,]
# train a naive bayes model
model <- NaiveBayes(Class~., data=data_train)
# make predictions
x_test <- data_test[,2:ncol(data_test)]
y_test <- data_test$Class
predictions <- predict(model, x_test)
# summarize results
confusionMatrix(predictions$class, y_test)

########## ENd of Naive Bayes ##############


######### Support Vector Machine ############
library(e1071)
#get data, split data into training and test, model with train, test on test
split=0.80
trainIndex <- createDataPartition(wine_data$Class, p=split, list=FALSE)
data_train <- wine_data[ trainIndex,]
data_test <- wine_data[-trainIndex,]
svm.model <- svm(Class ~ ., data = data_train,scale = TRUE, type = NULL,kernel = "radial" , cost = 80, gamma = 1)#vary cost gamma 
svm.pred <- predict(svm.model, data_test[,-1])
svm.model
svm.pred
t<-table(pred = svm.pred, true = data_test[,1])
t
acc=(sum(diag(t))/sum(t))*100 #accuracy calculation
acc
#############

#to play around with the parameters for more accuracy
#figure out why naive bayes accuracy higher than svm, when in fact svm is better classifier(random thoughts, to be verified)
#should any changes be done to the data before applying svm?? Very important!
#for tuning, to check out https://www.youtube.com/watch?v=ueKqDlMxueE
#for linear svm scale should be false
#still have to figure out how to plot the svm. 

#confusionMatrix(svm.pred$)
# x_test <- data_test[,2:ncol(data_test)]
# y_test <- data_test$Class
# #predictions <- predict(model, )
#
#
# svm.pred <- predict(svm.model,x_test)
# 
# confusionMatrix(predictio$class, y_test)

#alternative implementation of scm with self tuning of svm, here it automatically choose best parameters #96.667% accurate
 library(e1071)

sampp <- sample(1:nrow(iris), 0.80 * nrow(iris))

iristrain<-iris[sampp,]
iristest<-iris[-sampp,]

 tune <- tune.svm(Species~., 
                    data=iristrain, 
                    gamma=10^(-6:-1), 
                    cost=10^(1:4))
 summary(tune)
 
 model <- svm(Species~., 
              data=iristrain, 
              method="C-classification", 
              kernel="radial", 
              probability=T, 
              gamma=0.01, 
              cost=100)
 

 prediction <- predict(model, iristest)
 table(iristest$Species, prediction)
 pt<-table(iristest$Species, prediction)

 accuracyptron<-100*(sum(diag(pt))/sum(pt))
 accuracyptron
 #-------------------# 
 #testing on traing data alone#
data(iris)
svm.model <- svm(Species ~ Sepal.Length + Sepal.Width, data = iris, kernel = "linear")
# the + are support vectors
plot(iris$Sepal.Length, iris$Sepal.Width, col = as.integer(iris[, 5]), 
     pch = c("o","+")[1:150 %in% svm.model$index + 1], cex = 2, 
     xlab = "Sepal length", ylab = "Sepal width")

svm.pred  <- predict(svm.model, iris[,-5]) 
table(pred = svm.pred, true = iris[,5])
plot(svm.model, iris, Sepal.Width ~ Sepal.Length, 
     slice = list(sepal.width = 1, sepal.length = 2))

########################## End of SVM #############################



################## Neural Networks #############

#Neural network based on neural networks sandiego state university
#using adult data set from uci repository as it is more suitable for the neural network
source("http://scg.sdsu.edu/wp-content/uploads/2013/09/dataprep.r")
library(nnet)
a = nnet(income~., data=data$train,size=20,maxit=10000,decay=.001)
#decay is for decay of the weights 
#max it is the max number of iterations, usually it converges before reaching 10000
ress<-table(data$val$income,predict(a,newdata=data$val,type="class"))
accuracy<- sum(diag(ress))/sum(ress)
accuracy 
                    #########################
#to see if the model is better compared to random guessing we can use roc
library(ROCR)
pred = prediction(predict(a,newdata=data$val,type="raw"),data$val$income)
perf = performance(pred,"tpr","fpr")
plot(perf,lwd=2,col="blue",main="ROC - Neural Network  ")
abline(a=0,b=1)
## this is for one hidden layer only, 


#### Neural net based on  Matt Bogard tutorial on R Bloggers, #hidden = 0 makes it perceptron
require(neuralnet)
#nn <- neuralnet(case~age+parity+induced+spontaneous,data=infert, hidden=0, err.fct = "ce" ,linear.output=FALSE)
# The weight estimates can be obtained with the following command:

#nn$result.matrix
# And, the network can be plotted or visualized with the simple command:
#plot(nn)
######################3
#on iris dataset
library(neuralnet)

nnet_iristrain <-iris
sam<-0.80*nrow(iris)
samp<-120
nnet_iristrain<-iris[sample(1:nrow(iris), sam), ] 

 #Binarize the categorical output
nnet_iristrain <- cbind(nnet_iristrain, 
                            nnet_iristrain$Species == 'setosa')
nnet_iristrain <- cbind(nnet_iristrain,
                          nnet_iristrain$Species == 'versicolor')
nnet_iristrain <- cbind(nnet_iristrain, 
                          nnet_iristrain$Species == 'virginica')
names(nnet_iristrain)[6] <- 'setosa'
names(nnet_iristrain)[7] <- 'versicolor'
names(nnet_iristrain)[8] <- 'virginica'
nn <- neuralnet(setosa+versicolor+virginica ~ 
                    Sepal.Length+Sepal.Width
                  +Petal.Length
                  +Petal.Width,
                  data=nnet_iristrain, 
                  hidden = c(3,3)) 
plot(nn)
mypredict <- compute(nn, iris[-5])$net.result
# Put multiple binary output to categorical output
maxidx <- function(arr) {
    return(which(arr == max(arr)))
  }
idx <- apply(mypredict, c(1), maxidx)
prediction <- c('setosa', 'versicolor', 'virginica')[idx]
t<-table(prediction, iris$Species)
t
accurnn<-100*sum(diag(t))/sum(t)
accurnn
###################################### End of neural network ############


############### Perceptron  ######################
library(neuralnet)

perc_iristrain <-iris
sam<-0.80*nrow(iris) #for picking 80* sample
samp<-120
perc_iristrain<-iris[sample(1:nrow(iris), sam), ] 

#Binarize the categorical output
perc_iristrain <- cbind(perc_iristrain, 
                        perc_iristrain$Species == 'setosa')
perc_iristrain <- cbind(perc_iristrain,
                        perc_iristrain$Species == 'versicolor')
perc_iristrain <- cbind(perc_iristrain, 
                        perc_iristrain$Species == 'virginica')
names(perc_iristrain)[6] <- 'setosa'
names(perc_iristrain)[7] <- 'versicolor'
names(perc_iristrain)[8] <- 'virginica'
nn <- neuralnet(setosa+versicolor+virginica ~ 
                  Sepal.Length+Sepal.Width
                +Petal.Length
                +Petal.Width,
                data=perc_iristrain, 
                hidden = 0,threshold = 0.001,stepmax = 1e+05,rep=5,startweights = NULL, 
                learningrate.limit = NULL, 
                learningrate.factor = list(minus = 0.5, plus = 1.2), 
                learningrate=NULL, lifesign = "full", 
                lifesign.step = 1000, algorithm = "rprop+", 
                err.fct = "sse", act.fct = "logistic", 
                linear.output = TRUE, exclude = NULL, 
                constant.weights = NULL, likelihood = FALSE) 
plot(nn)
mypredict <- compute(nn, iris[-5])$net.result
# Put multiple binary output to categorical output
maxidx <- function(arr) {
  return(which(arr == max(arr)))
}
idx <- apply(mypredict, c(1), maxidx)
prediction <- c('setosa', 'versicolor', 'virginica')[idx]
t<-table(prediction, iris$Species)
t
accurnn<-100*sum(diag(t))/sum(t)
accurnn

####################3 End of perceptron##################

#reference: 
# https://horicky.blogspot.pt/2012/06/predictive-analytics-neuralnet-bayesian.html
# http://www.statmethods.net/advstats/cart.html
# http://www.parallelr.com/r-deep-neural-network-from-scratch/
#   http://scg.sdsu.edu/ann_r/
