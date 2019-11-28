
#Set working directory
setwd("C:\\Users\\PHANI KUMAR\\Desktop\\Machine Learning Projects\\3. BANK MARKETING CASE STUDY - CLASSIFICATION")

#Reading the data
traindata <- read.csv("Train_nyOWmfK.csv")

#Loading required packages
library(lubridate)
library(car)
library(caret)
library(flipTime)
library(InformationValue)
library(ROCR)
library(pROC)
library(ROSE)
options(scipen = 999)

#Understanding the data
summary(traindata)
str(traindata)

#Removing unnecessary columns
traindata$Var1 <- NULL
traindata$Var2 <- NULL
traindata$Var4 <- NULL
traindata$Var5 <- NULL

class(traindata$DOB)

#Converting date colums to proper format

traindata$DOB <- as.character(traindata$DOB)
dt <- as.Date(traindata$DOB,format = "%d-%b-%y")
traindata$DOB <- as.Date(format(dt,"19%y-%m-%d"))


traindata$Lead_Creation_Date <- as.character(traindata$Lead_Creation_Date)
dt2 <- as.Date(traindata$Lead_Creation_Date,format = "%d-%b-%y")
traindata$Lead_Creation_Date <- as.Date(format(dt2,"20%y-%m-%d"))


#Deriving age column
traindata$Age <- traindata$Lead_Creation_Date-traindata$DOB
traindata$Age <- traindata$Age/365.25
traindata$Age <- as.numeric(round(traindata$Age,0))

#Removing unnecessary columns
traindata$DOB <- NULL
traindata$Lead_Creation_Date <- NULL
traindata$Source <- NULL

#Missing values in the data

colSums(is.na(traindata))

#Imputing missing data manually
traindata$Loan_Amount_Applied[is.na(traindata$Loan_Amount_Applied)] <- median(traindata$Loan_Amount_Applied,na.rm = T)
traindata$Loan_Tenure_Applied[is.na(traindata$Loan_Tenure_Applied)] <- median(traindata$Loan_Tenure_Applied,na.rm = T)
traindata$Existing_EMI[is.na(traindata$Existing_EMI)] <- mean(traindata$Existing_EMI,na.rm = T)
traindata$Loan_Amount_Submitted[is.na(traindata$Loan_Amount_Submitted)] <- median(traindata$Loan_Amount_Submitted,na.rm = T)
traindata$Loan_Tenure_Submitted[is.na(traindata$Loan_Tenure_Submitted)] <- median(traindata$Loan_Tenure_Submitted,na.rm = T)
traindata$Interest_Rate[is.na(traindata$Interest_Rate)] <- median(traindata$Interest_Rate,na.rm = T)
traindata$Processing_Fee[is.na(traindata$Processing_Fee)] <- median(traindata$Processing_Fee,na.rm = T)
traindata$EMI_Loan_Submitted[is.na(traindata$EMI_Loan_Submitted)] <- median(traindata$EMI_Loan_Submitted,na.rm = T)

colSums(is.na(traindata))

#User defined functio to get descriptives

my_stats <- function(x){
  if(class(x)=="numeric"|class(x)=="integer"){
    Var_Type=class(x)
    n<-length(x)
    nmiss<-sum(is.na(x))
    miss_pct <- nmiss/length(x)*100
    mean<-mean(x,na.rm=T)
    median <- median(x,na.rm = T)
    std<-sd(x,na.rm=T)
    var<-var(x,na.rm=T)
    min<-min(x,na.rm=T)
    pctl <- quantile(x, na.rm=T, p=c(0.01,0.05,0.1,0.25,0.5,0.75,0.9, 0.95,0.99,1.00))
    max<-max(x,na.rm=T)
    return(c(Var_Type=Var_Type, n=n,nmiss=nmiss,miss_pct=miss_pct,mean=mean,median=median,std=std,var=var,min=min,pctl=pctl))
  }
  else{
    Var_Type=class(x)
    n<-length(x)
    nmiss<-sum(is.na(x))
    fre<-table(x)
    prop<-prop.table(table(x))
    
    return(c(Var_Type=Var_Type, n=n,nmiss=nmiss,freq=fre,proportion=prop))
  }
}

#Splitting the data into numeric and categorical data

num_vars <- sapply(traindata,is.numeric)
cat_vars <- !sapply(traindata,is.numeric)

#Preparing descriptive stats

numeric_traindata_stats <- data.frame(apply(traindata[num_vars],2,my_stats))

write.csv(numeric_traindata_stats,"Descriptive stats.csv")

categorical_traindata_stats <- apply(traindata[cat_vars],2,my_stats)

traindata_num <- data.frame(traindata[num_vars])
traindata_cat <- data.frame(traindata[cat_vars])

#No missing data so we can proceed with outlier treatment
#UDF for outlier treatment

outlier_treat <- function(x){
  UC1 = quantile(x, p=0.99,na.rm=T)
  LC1 = quantile(x, p=0.01,na.rm=T)
  x=ifelse(x>UC1, UC1, x)
  x=ifelse(x<LC1, LC1, x)
  return(x)
  
}

#applying outlier treatment for numeric data frame

traindata_num <- apply(traindata_num,2,FUN = outlier_treat)
traindata_num <- data.frame(traindata_num)


#Correlation matrix for numeric variables

correlation_matrix <- data.frame(cor(traindata_num,method = "pearson"))


#Checking for Significant Categorical Variables
traindata_cat$Disbursed <- traindata_num$Disbursed

summary(aov(Disbursed~Gender,data = traindata_cat))
summary(aov(Disbursed~City,data = traindata_cat))
summary(aov(Disbursed~Mobile_Verified,data = traindata_cat))
summary(aov(Disbursed~Filled_Form,data = traindata_cat))
summary(aov(Disbursed~Device_Type,data = traindata_cat))
summary(aov(Disbursed~Salary_Account,data = traindata_cat))
summary(aov(Disbursed~Employer_Name,data = traindata_cat))


#The following variables are significant
#Gender
#Mobile_Verified
#Filled_Form
#Device_Type
#Salary_Account

#Creating dummy variables for the above significant variables
#Salary account has 48 or more levels. so its better t leave that variable.

dv1 <- caret::dummyVars(~Gender,traindata_cat)
Dummy_Gender = data.frame(predict(dv1, traindata_cat))[-1]

dv2 <- caret::dummyVars(~Mobile_Verified,traindata_cat)
Dummy_Mobile = data.frame(predict(dv2, traindata_cat))[-1]

dv3 <- caret::dummyVars(~Filled_Form,traindata_cat)
Dummy_Filled = data.frame(predict(dv3, traindata_cat))[-1]

dv4 <- caret::dummyVars(~Device_Type,traindata_cat)
Dummy_Device = data.frame(predict(dv4, traindata_cat))[-1]

#Creating finaldata
trainingdata <- cbind(traindata_num,Dummy_Gender,Dummy_Mobile,Dummy_Filled,Dummy_Device)

#splitting the data into training and testing

set.seed(1010)
train_ind <- sample(1:nrow(trainingdata), size = floor(0.70 * nrow(trainingdata)))

training <- trainingdata[train_ind,]
testing <- trainingdata[-train_ind,]

#Building binomial logistic regression model for training data

model_1 <- glm(Disbursed~.,data = training,family = binomial(logit))

summary(model_1)

model_1$coefficients

#Stepwise regression of the above model

step_resgression <- step(model_1,direction = "both")

summary(step_resgression)

#Building final model on the above stepAIC

final_model <- glm(Disbursed~Monthly_Income+Existing_EMI+Interest_Rate+LoggedIn+
                     Mobile_Verified.Y+Filled_Form.Y,data = training,family = binomial(link = "logit"))

summary(final_model)

summary(step(final_model))

#############################Predictions and Evaluation Metrics for Training dataset########################

train_evaluation <- cbind(training, Prob=predict(final_model, type="response")) 

cut_off <- optimalCutoff(train_evaluation$Disbursed, train_evaluation$Prob,
                         optimiseFor = "Both", returnDiagnostics = TRUE)

cut_off$sensitivityTable
cut_off$Specificity
cut_off$TPR
cut_off$FPR

#Concordance
Concordance(train_evaluation$Disbursed,train_evaluation$Prob)

roc_table <- data.frame(cut_off$sensitivityTable)

#Somers distance
somersD(train_evaluation$Disbursed, train_evaluation$Prob)

#Confusion matrix
confusion_matrix <- confusionMatrix(train_evaluation$Disbursed,train_evaluation$Prob, threshold=0.26)

write.csv(confusion_matrix,"Confusion Matrix.csv")

#roc curve
pROC::roc(train_evaluation$Disbursed, train_evaluation$Prob)

#Area under the roc curve
AUROC(train_evaluation$Disbursed, train_evaluation$Prob)

#Plotting ROC curve
plotROC(train_evaluation$Disbursed, train_evaluation$Prob, Show.labels=F)

#Roc and Auc plot 
roc.curve(train_evaluation$Disbursed, train_evaluation$Prob, plotit = T)

#############################Predictions and Evaluation Metrics for testing dataset########################

test_evaluation <- cbind(testing, Prob=predict(final_model,testing,type="response")) 

cut_off2 <- optimalCutoff(test_evaluation$Disbursed, test_evaluation$Prob,
                         optimiseFor = "Both", returnDiagnostics = TRUE)

cut_off2$sensitivityTable
cut_off2$Specificity
cut_off2$TPR
cut_off2$FPR

#Concordance
Concordance(test_evaluation$Disbursed,test_evaluation$Prob)

roc_table2 <- data.frame(cut_off$sensitivityTable)

#Somers distance
somersD(test_evaluation$Disbursed, test_evaluation$Prob)

#Confusion matrix
confusion_matrix_test <- confusionMatrix(test_evaluation$Disbursed,test_evaluation$Prob, threshold=0.26)

write.csv(confusion_matrix_test,"Confusion Matrix Testing.csv")

#roc curve
pROC::roc(test_evaluation$Disbursed, test_evaluation$Prob)

#Area under the roc curve
AUROC(test_evaluation$Disbursed, test_evaluation$Prob)

#Plotting ROC curve
plotROC(test_evaluation$Disbursed, test_evaluation$Prob, Show.labels=F)

#Roc and Auc plot 
roc.curve(test_evaluation$Disbursed, test_evaluation$Prob, plotit = T)

##################################################End of classification###############################















































