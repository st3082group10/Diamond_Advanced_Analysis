dataset = read.csv("diamonds.csv")
dataset$cut = factor(dataset$cut , levels=c("Fair", "Good", "Very Good", "Premium", "Ideal"))
dataset$clarity = factor(dataset$clarity , levels=c("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"))
dataset$color = factor(dataset$color , levels=c("J","I","H","G","F","E","D"))

head(dataset)

# Cleaning the dataset

# Removing unusual observations as x, y, z cannot take 0 values
dataset = dataset[!(dataset$x == 0 | dataset$y == 0 | dataset$z == 0),]

# We can remove incorrect data entering using depth variable as depth is calculated using x,y,z 
depth2 = numeric(0)
for (i in 1:nrow(dataset)) {
  depth2[i] = (dataset$z[i]/mean(c(dataset$x[i], dataset$y[i])))*100
}

df = dataset[abs(dataset$depth - depth2) > 1,]
nrow(df)

dataset = dataset[!(abs(dataset$depth - depth2) > 1),]

# Applying transformations to reduce skewness
dataset$ln_price = log(dataset$price)
dataset$ln_carat = log(dataset$carat)


# Splitting the dataset into training and test data
set.seed(123)
test_index = sample(1:nrow(dataset), nrow(dataset)*0.2)
test = dataset[test_index,-1]
train = dataset[-test_index,-1]

# Multiple linear regression

library(Metrics)
library(car)
library(sjPlot)
library(sjmisc)
library(sjlabelled)
library(webshot)
library(MLmetrics)
library(ggplot2)


mls_model = lm(ln_price~ln_carat+color+cut+clarity , data = train)
summary(mls_model)
tab_model(mls_model, show.se = TRUE, show.ci = F, file = "coeftable.html", digits = 4)
webshot("coeftable.html", "coeftable.png")

# Training and testing evaluation (ln_price as the response)

# Training
rmse(train$ln_price, mls_model$fitted.values)
mape(train$ln_price, mls_model$fitted.values)*100

# Test 
test$pred_lnprice = predict(mls_model, test)
rmse(test$ln_price, test$pred_lnprice)
mape(test$ln_price, test$pred_lnprice)*100

test$price_pred = exp(predict(mls_model, newdata = test))

# Checking the assumptions 

# Normality
model_residuals = mls_model$residuals
hist(model_residuals, breaks = 50)


qqnorm(model_residuals)
# Plot the Q-Q line
qqline(model_residuals)

# Autocorrelation
durbinWatsonTest(mls_model)


# Multicollinearity
vif(mls_model)

plot(mls_model$fitted.values, mls_model$residuals, main="Residual vs fitted values",
     xlab="Fitted values", ylab="Residuals", pch=19)

plot(mls_model, which = 1)
plot(mls_model, which = 4)
plot(mls_model, which = 5)

# Constant variance
library(ggfortify)

autoplot(mls_model, which = 3, label.size = 2, ncol = 1)+theme_light()
ncvTest(mls_model)

lmtest::bgtest(mls_model)

# Actual price vs pred price

ggplot(test, aes(x=ln_price, y=pred_lnprice)) + 
  geom_point(size = 2, color = "#065b58")+
  xlab("Actual ln(Price)")+
  ylab("Predicted ln(Price)")+theme_light()

ggplot(test, aes(x=price, y=price_pred)) + 
  geom_point(size = 2, color = "#065b58")+
  xlab("Actual Price")+
  ylab("Predicted Price")+
  theme_light()

ggplot(test, aes(x=carat)) + 
  geom_point(aes(y = price),size = 2, color = "#123f51")+
  geom_point(aes(y = price_pred),size = 2, color = "#3a8692")+
  xlab("Carat")+
  ylab("Price")+
  theme_light()


vals = data.frame(carat = rep(test$carat, times = 2), price = c(test$price, test$price_pred), grp = c(rep("Actual", times = 10768), rep("Predicted", 10768)))
cols = c("#123f51", "#3a8692")
ggplot(data = vals, aes(x = carat, y = price, color = grp))+
  geom_point(size = 2)+
  scale_color_manual(values = cols)+
  labs(color = NULL)+
  theme_light()+
  theme(legend.text = element_text(size = 12))


# Predicting price by taking exp(ln_price)

# Train
rmse(train$price,exp(mls_model$fitted.values))
mape(train$price, exp(mls_model$fitted.values))*100

# Test

rmse(test$price, test$price_pred)
mape(test$price, test$price_pred)*100


cor(test$price, test$price_pred)

plot(test$price,test$price_pred)


# Regularization techniques

library(magrittr)
library(dplyr)
library(caret)
library(glmnet)
library(Metrics)

X_train = train[, c(1:6, 8:10)]
X_train$cut = as.numeric(X_train$cut)
X_train$clarity = as.numeric(X_train$clarity)
X_train$color = as.numeric(X_train$color)

X_test = test[, c(1:6, 8:10)]
X_test$cut = as.numeric(X_test$cut)
X_test$clarity = as.numeric(X_test$clarity)
X_test$color = as.numeric(X_test$color)
y_train = train[,7]
y_test = test[,7]

#Ridge
ridge = train(y = y_train,
              x = X_train,
              method = 'glmnet', 
              tuneGrid = expand.grid(alpha = 0, lambda = 1)
              
)
# Make the predictions
predictions_ridge = ridge %>% predict(X_test)

# Print R squared scores
data.frame(
  Ridge_R2 = R2(predictions_ridge, y_test))

# Print RMSE
data.frame(
  Ridge_RMSE = RMSE(predictions_ridge, y_test)
)

#lasso
lasso = train(y = y_train,
              x = X_train,
              method = 'glmnet', 
              tuneGrid = expand.grid(alpha = 1, lambda = 1)
              
) 
# Make the predictions
predictions_lasso = lasso %>% predict(X_test)

# Print R squared scores
data.frame(
  Lasso_R2 = R2(predictions_lasso, y_test))

# Print RMSE
data.frame(
  Lasso_RMSE = RMSE(predictions_lasso, y_test)
)
# Print coeficients
data.frame(
  
  as.data.frame.matrix(coef(lasso$finalModel, lasso$bestTune$lambda))
)

parameters = c(seq(0.1, 2, by =0.1) ,  seq(2, 5, 0.5) , seq(5, 25, 1))
lasso<-train(y = y_train,
             x = X_train,
             method = 'glmnet', 
             tuneGrid = expand.grid(alpha = 1, lambda = parameters),
             metric =  "Rsquared"
             
) 

#Elasticnet
elastic = train(y = y_train,
                x = X_train,
                method = 'glmnet', 
                tuneGrid = expand.grid(alpha = 0.5, lambda = 1)
                
) 
# Make the predictions
predictions_elastic = elastic %>% predict(X_test)

# Print R squared scores
data.frame(
  Elastic_R2 = R2(predictions_elastic, y_test))

# Print RMSE
data.frame(
  Elastic_RMSE = RMSE(predictions_elastic, y_test)
)

linear = train(y = y_train, 
               x = X_train, 
               method = 'lm',
               metric =  "Rsquared"
)

print(paste0('Ridge best parameters: ' , ridge$finalModel$lambdaOpt))

predictions_ridge <- ridge %>% predict(X_test)
predictions_lin <- linear %>% predict(X_test)
predictions_lasso <- lasso%>%predict(X_test)
predictions_elastic<- elastic%>%predict(X_test)

predictions_ridge1 <- ridge %>% predict(X_train)
predictions_lin1 <- linear %>% predict(X_train)
predictions_lasso1 <- lasso%>%predict(X_train)
predictions_elastic1<- elastic%>%predict(X_train)

data.frame(
  Ridge_R2 = R2(predictions_ridge, y_test),
  Linear_R2 = R2(predictions_lin, y_test),
  Lasso_R2 = R2(predictions_lasso,y_test),
  Elastic_R2 = R2(predictions_elastic, y_test)
)

data.frame(
  Ridge_RMSE = RMSE(predictions_ridge, y_test) , 
  Linear_RMSE = RMSE(predictions_lin, y_test),
  Lasso_RMSE = RMSE(predictions_lasso, y_test),
  Elastic_RMSE = RMSE(predictions_elastic, y_test)
)

data.frame(
  Ridge_mape = mape(y_test, predictions_ridge)*100 , 
  Lasso_mape = mape(y_test, predictions_lasso)*100,
  Elastic_mape = mape(y_test, predictions_elastic)*100)

data.frame(
  Ridge_RMSE = RMSE(predictions_ridge1, y_train) , 
  Linear_RMSE = RMSE(predictions_lin1, y_train),
  Lasso_RMSE = RMSE(predictions_lasso1, y_train),
  Elastic_RMSE = RMSE(predictions_elastic1, y_train)
)

data.frame(
  Ridge_mape = mape(y_train, predictions_ridge1)*100 , 
  Lasso_mape = mape(y_train, predictions_lasso1)*100,
  Elastic_mape = mape(y_train, predictions_elastic1)*100)

print('Best estimator coefficients')

data.frame(
  ridge = as.data.frame.matrix(coef(ridge$finalModel, ridge$finalModel$lambdaOpt)),
  linear = (linear$finalModel$coefficients)
)

# Set lambda coefficients
paramRidge = seq(0, 1000, 10)
paramLasso = seq(0, 1000, 10)
paramElastic = seq(0, 1000, 10)


# Convert X_train to matrix for using it with glmnet function
X_train_m = as.matrix(X_train)

# Build Ridge and Lasso for 200 values of lambda 
rridge = glmnet(
  x = X_train_m,
  y = y_train,
  alpha = 0, #Ridge
  lambda = (paramRidge)
  
)
llasso = glmnet(
  x = X_train_m,
  y = y_train,
  alpha = 1, #Lasso
  lambda = (paramLasso)
  
)
eelastic = glmnet(
  x = X_train_m,
  y = y_train,
  alpha = 0.5, #Ridge
  lambda = (paramElastic)
  
)

plot(rridge, xvar = 'lambda', label = T)
plot(rridge, xvar = 'dev', label = T)
plot(llasso, xvar = 'lambda', label = T)
plot(llasso, xvar = 'dev', label = T)
plot(eelastic, xvar = 'lambda', label = T)
plot(eelastic,xvar = 'dev', label = T)
tail(predictions_ridge)
head(predictions_ridge)

head(predictions_lin)


# Random forest

training = train[, c(1:6, 8:10, 7)]
testing=test[,c(1:6, 8:10, 7)]

## Random Forest Model ##

library(randomForest)

#Train the model
set.seed(123)
model=randomForest(price~.,data=training)
model

# Evaluating Model Accuracy
price_pred = predict(model,testing)
testing$price_pred=price_pred
result=View(testing[(1:20),])

# visualization of price vs price_pred of testing set according to carat values
library(ggplot2)

# Build scatterplot
ggplot(  ) + 
  geom_point( aes(x = testing$carat, y =test$price, color = 'red', alpha = 0.5) ) + 
  geom_point( aes(x = testing$carat , y = price_pred, color = 'blue',  alpha = 0.5)) + 
  labs(x = "Carat", y = "Price", color = "", alpha = 'Transperency') +
  scale_color_manual(labels = c( "Predicted", "Real"), values = c("blue", "red")) 


##precision, recall
print(paste0('MAE: ' , mae(test$price,price_pred) ))
print(paste0('MSE: ' ,caret::postResample(price_pred , testing$price)['RMSE']^2 ))
print(paste0('R2: ' ,caret::postResample(price_pred , testing$price)['Rsquared'] ))

# Test data
rmse(testing$price, price_pred)
mape(testing$price, price_pred)*100
cor(testing$price, price_pred)

ggplot() + 
  geom_point( aes(x = test$price, y =price_pred) ) + 
  labs(x = "Actual", y = "Predicted") 

# Training data

rmse(training$price, model$predicted)
mape(training$price, model$predicted)*100


#Feature Importance

varImpPlot(model, main ='Feature importance')
