library(Quandl)
library(forecast)
library("forecTheta")
library(xts)
library("rlist")
Quandl.api_key("insert your key here")

# from FRED 
df <- read.csv("FRED-datasets-codes.csv", header=TRUE, stringsAsFactors = FALSE)

names_lim <- df$code[1:40000]

quandldownload <- function(names){
  yearly <- list()
  monthly <- list()
  quatly <- list()
  for(name in 1:length(names)){
    Sys.sleep(0.5)
    #ERROR HANDLING
    tryCatch(
      {series = Quandl(names[name], type="ts")
      period <- periodicity(series)$scale
      if (period == "yearly") {
        yearly <- list.append(yearly, series)
      }else if(period == "quarterly") {
        quatly <- list.append(quatly, series)
      }else if(period == "monthly"){
        monthly <- list.append(monthly, series)
      }
      },
      warning = function(w) {name=name+1},
      error = function(e) {name=name+1}
    )
    
  }
  return(list("yearly"=yearly,"quatly"=quatly,"monthly"=monthly))
}

# итоговый список рядов по частотам
list_of_series <- quandldownload(names_lim)
saveRDS(list_of_series,"first40k_new")
# 141 000
file <- readRDS('first40k_new')
a <- c()
for(i in 1:length(file$quatly)){
  if (anyNA(file$quatly[[i]])== TRUE){
    a <- c(a,i)
  } #  2105 рядов с пропусками 
}
length(a) # список индексов рядов с пропусками
clean_quatly <- file$quatly[-a]  ##### очищенный список квартальных рядов
length(clean_quatly)


a <- c()
for(i in 1:length(file$monthly)){
  if (anyNA(file$monthly[[i]])== TRUE){
    a <- c(a,i)
  } #  2105 рядов с пропусками 
}
length(a) # список индексов рядов с пропусками
clean_monthly <- file$monthly[-a]  ##### очищенный список месячных рядов
length(clean_monthly)


a <- c()
for(i in 1:length(file$yearly)){
  if (anyNA(file$yearly[[i]])== TRUE){
    a <- c(a,i)
  } #  2105 рядов с пропусками 
}
length(a) # список индексов рядов с пропусками
clean_yearly <- file$yearly[-a]  ##### очищенный список годовых рядов
length(clean_yearly)



saveRDS(clean_monthly,'monthly_data')
saveRDS(clean_yearly,'yearly_data')
saveRDS(clean_quatly,'quatly_data')

## КВАРТАЛЬНЫЕ ДАННЫЕ
horq=4
qu <- readRDS('quatly_data')[1:2000] # возьмем выборку поменьше
qu1 <- qu[lengths(qu)>horq+3]
length(qu1) # исходные квартальные данные без очень коротких рядов 

## Обрежем ряды до 50 наблюдений
qu2 <- list()
for(i in 1:length(qu1)){
  if (length(qu1[[i]])<= 50){
    qu2 <- list.append(qu2, qu1[[i]])
  }else{
    short_series <- window(qu1[[i]],start = time(qu1[[i]])[1],end=time(qu1[[i]])[50])
    qu2 <- list.append(qu2, short_series)
  }}
length(qu2) # исходные укороченные до 50 наблюдений квартальные данные без очень коротких рядов -- их использовать в качестве a_true

saveRDS(qu2,"shorten_quarter_data_1999")

b <- c()
for(i in 29:length(qu1)){
  print(i)
  fit <- tbats(qu1[[i]])
  if (!is.null(fit$seasonal)== TRUE){
    b <- c(b,i)
  }  
}
length(b) 
saveRDS(b,"seasonal_quarter_index_new")

# additive for quarter data -- делаем вектор сезонных частей, неукороченный до 50 наблюдений
b <- readRDS("seasonal_quarter_index_new")
seas_qu <- list()
for(i in 1:length(qu1)){
  if(i %in% b == TRUE){
    season <- decompose(qu1[[i]],"additive")$seasonal
  }else{
    season <- qu1[[i]] - qu1[[i]]
  }
  seas_qu <- list.append(seas_qu, season)
} 

# теперь укоротим сезонную часть до 50 наблюдений
seas_qu2 <- list()
for(i in 1:length(seas_qu)){
  if (length(seas_qu[[i]])<= 50){
    seas_qu2 <- list.append(seas_qu2, seas_qu[[i]])
  }else{
    short_series <- window(seas_qu[[i]],start = time(seas_qu[[i]])[1],end=time(seas_qu[[i]])[50])
    seas_qu2 <- list.append(seas_qu2, short_series)
  }}
length(seas_qu2)

saveRDS(seas_qu2, "seasonalpart_shorten_quarter_data")


# создадим десезоннированую квартальную выборку (укороченную)
deseas_qu <- list()
for(i in 1:length(qu2)){
  myser <- qu2[[i]] - seas_qu2[[i]]
  deseas_qu <- list.append(deseas_qu, myser)
}
length(deseas_qu)
saveRDS(deseas_qu, "shorten_deseasonalized_quarter_data")


#######################################################################################################
## МЕСЯЧНЫЕ ДАННЫЕ
horm=6
mon <- readRDS('monthly_data')[1:2000] # возьмем выборку поменьше
mon1 <- mon[lengths(mon)>horm+3]
length(mon1) # исходные месячные данные без очень коротких рядов -- их 2000

## Обрежем ряды до 50 наблюдений
mon2 <- list()
for(i in 1:length(mon1)){
  if (length(mon1[[i]])<= 50){
    mon2 <- list.append(mon2, mon1[[i]])
  }else{
    short_series <- window(mon1[[i]],start = time(mon1[[i]])[1],end=time(mon1[[i]])[50])
    mon2 <- list.append(mon2, short_series)
  }}
length(mon2) # исходные укороченные до 50 наблюдений месячные данные без очень коротких рядов 
saveRDS(mon2,"shorten_month_data_2000")



d <- c()
for(i in 1:length(mon1)){
  print(i)
  fit <- tbats(mon1[[i]])
  if (!is.null(fit$seasonal)== TRUE){
    d <- c(d,i)
  }  
}
length(d) 
saveRDS(d,"seasonal_month_index_new")

# multiplicative for month data -- делаем вектор сезонных частей, неукороченный до 50 наблюдений
d <- readRDS("seasonal_month_index_new")
seas_mon <- list()
for(i in 1:length(mon1)){
  if(i %in% d == TRUE){
    season <- decompose(mon1[[i]],"multiplicative")$seasonal
  }else{
    season <- mon1[[i]] / mon1[[i]]
  }
  seas_mon <- list.append(seas_mon, season)
} 

length(seas_mon)
# теперь укоротим сезонную часть до 50 наблюдений
seas_mon2 <- list()
for(i in 1:length(seas_mon)){
  if (length(seas_mon[[i]])<= 50){
    seas_mon2 <- list.append(seas_mon2, seas_mon[[i]])
  }else{
    short_series <- window(seas_mon[[i]],start = time(seas_mon[[i]])[1],end=time(seas_mon[[i]])[50])
    seas_mon2 <- list.append(seas_mon2, short_series)
  }}
length(seas_mon2)

saveRDS(seas_mon2, "seasonalpart_shorten_month_data") 
length(seas_mon2[[1]])

# создадим десезоннированую месячную выборку (укороченную)


mon2 <- readRDS("shorten_month_data_2000")
seas_mon2<- readRDS("seasonalpart_shorten_month_data")
deseas_mon <- list()
for(i in 1:length(mon2)){
  myser <- mon2[[i]] / seas_mon2[[i]]
  deseas_mon <- list.append(deseas_mon, myser)
}
length(deseas_mon)
saveRDS(deseas_mon, "shorten_deseasonalized_month_data")

#######################################################################################################
## ГОДОВЫЕ ДАННЫЕ
hory=3
ye <- readRDS('yearly_data')[1:2000] # возьмем выборку поменьше
ye1 <- ye[lengths(ye)>hory+3]
length(ye1) # исходные годовые данные без очень коротких рядов -- их

## Обрежем ряды до 50 наблюдений
ye2 <- list()
for(i in 1:length(ye1)){
  if (length(ye1[[i]])<= 50){
    ye2 <- list.append(ye2, ye1[[i]])
  }else{
    short_series <- window(ye1[[i]],start = time(ye1[[i]])[1],end=time(ye1[[i]])[50])
    ye2 <- list.append(ye2, short_series)
  }}
length(ye2) # исходные укороченные до 50 наблюдений годовые данные без очень коротких рядов -- их использовать в качестве a_true

saveRDS(ye2,"shorten_year_data_2000")


# Forecasting horizon: yearly(3), quatly(4), monthly(6)
hory <- 3
horq <- 4
horm <- 6

#COUNTING ERROR 

metrics <- function(data,hor){
  ymase <- data.frame(arima=double(),arfima=double(),ets=double(),tbats=double(),meanf=double(),naive=double(),
                      rwf=double(),snaive=double(),rwdf=double(),theta=double(),otm=double(),dotm=double(),stringsAsFactors=FALSE)
  colnames(ymase) <- c('arima','arfima','ets','tbats','meanf','naive','rwf','snaive','rwdf','theta','otm','dotm')
  ysmdape<- data.frame(arima=double(),arfima=double(),ets=double(),tbats=double(),meanf=double(),naive=double(),
                       rwf=double(),snaive=double(),rwdf=double(),theta=double(),otm=double(),dotm=double(),stringsAsFactors=FALSE)
  colnames(ysmdape) <- c('arima','arfima','ets','tbats','meanf','naive','rwf','snaive','rwdf','theta','otm','dotm')
  for(series in 1:length(data))  {
    a <- data[[series]]
    print(series)
    a_train <- forecast:::subset.ts(a,end=length(a)-hor)
    a_test <- forecast:::subset.ts(a, start = length(a)-hor+1)
    # # METHODS
    tryCatch(pred_arfima <- forecast:::forecast(forecast:::arfima(a_train), h=hor)$mean, error = function(e) {pred_arfima <- NA})
    pred_arima <- forecast:::forecast(forecast:::auto.arima(a_train,ic='aicc'),h=hor)$mean
    tryCatch(pred_ets <- forecast:::forecast(forecast:::ets(a_train),h=hor)$mean,error = function(e) {pred_ets <- NA})
    tryCatch(pred_tbats <- forecast:::forecast(forecast:::tbats(a_train),h=hor)$mean,error = function(e) {pred_tbats <- NA})
    tryCatch(pred_mean <- forecast:::meanf(a_train, h = hor)$mean,error = function(e) {pred_mean <- NA})
    tryCatch(pred_naive <- forecast:::naive(a_train, h = hor)$mean,error = function(e) {pred_naive <- NA})
    tryCatch(pred_rwf <- forecast:::rwf(a_train, h = hor)$mean,error = function(e) {pred_rwf <- NA})
    tryCatch(pred_snaive <- forecast:::snaive(a_train, h=hor)$mean,error = function(e) {pred_snaive <- NA})
    tryCatch(pred_rwdf <- forecast:::rwf(a_train, drift = TRUE, h=hor)$mean,error = function(e) {pred_rwdf <- NA})
    tryCatch(pred_theta <- forecast:::thetaf(a_train, h=hor)$mean,error = function(e) {pred_theta <- NA})
    tryCatch(pred_otm <- summary(forecTheta:::otm(a_train, h=hor))$statistics[1:hor],error = function(e) {pred_otm <- NA})
    tryCatch(pred_dotm <- summary(forecTheta:::dotm(a_train, h=hor))$statistics[1:hor],error = function(e) {pred_dotm <- NA})
    #PREDICTIONS & ERROR
    mase <- c()
    smdape <- c()
    for (pred in list(pred_arima,pred_arfima,pred_ets,pred_tbats,pred_mean,
                      pred_naive,pred_rwf,pred_snaive,pred_rwdf,pred_theta,pred_otm,pred_dotm)){
      mase <- c(mase, ftsa:::error(forecast=pred, true=a_test,insampletrue=a_train, method='mase'))
      smdape <- c(smdape, ftsa:::error(forecast=pred, true=a_test,insampletrue=a_train, method='smdape'))
    }
    ymase[nrow(ymase)+1,] <- mase
    ysmdape[nrow(ysmdape)+1,]  <- smdape
  }
  return(list("ymase"=ymase,"ysmdape"=ysmdape)) #"ymdase"=ymdase,"ysmape"=ysmape, 'yrmsse'=yrmsse,
}

year_metrics <- metrics(ye2,hory)
saveRDS(year_metrics, "year_list_of_errors_new")
year_metrics <- readRDS("year_list_of_errors_new")
## считаем средние ошибки
ymase <- year_metrics$ymase
ymase[!is.finite(as.matrix( year_metrics$ymase))] <- NA
colMeans(ymase, na.rm=TRUE) # таблица ошибок прогнозов алгоритмов

ysmdape <- year_metrics$ysmdape
ysmdape[!is.finite(as.matrix(year_metrics$ysmdape))] <- NA
colMeans(ysmdape, na.rm=TRUE) # таблица ошибок прогнозов алгоритмов

qu2 <- readRDS("shorten_quarter_data_1999")
quarter_metrics <- metrics(qu2,horq)
saveRDS(quarter_metrics, "quarter_list_of_errors_new")

ymase <- quarter_metrics$ymase
ymase[!is.finite(as.matrix(ymase))] <- NA
colMeans(ymase, na.rm=TRUE) # таблица ошибок прогнозов алгоритмов 

ysmdape <- quarter_metrics$ysmdape
ysmdape[!is.finite(as.matrix(ysmdape))] <- NA
colMeans(ysmdape, na.rm=TRUE) # таблица ошибок прогнозов алгоритмов

mon2 <- readRDS("shorten_month_data_2000")
month_metrics <- metrics(mon2,horm)
saveRDS(month_metrics, "month_list_of_errors_new")

ymase <- month_metrics$ymase
ymase[!is.finite(as.matrix(ymase))] <- NA
colMeans(ymase, na.rm=TRUE) # таблица ошибок прогнозов алгоритмов 

ysmdape <- month_metrics$ysmdape
ysmdape[!is.finite(as.matrix(ysmdape))] <- NA
colMeans(ysmdape, na.rm=TRUE) # таблица ошибок прогнозов алгоритмов 


# ERROR OF NEW METHOD
##### YEARLY DATA
##--------------------------------------------------------------------------------------    

list_of_metr <- data.frame(mase=double(),mdase=double(),smape=double(),rmsse=double(),smdape=double(),stringsAsFactors=FALSE) 
colnames(list_of_metr) <- c('mase','mdase','smape','rmsse','smdape')  
##--------------------------------------------------------------------------------------    
ye2 <- readRDS("shorten_year_data_2000") ## истинные укороченные ряды

predictions <- read.csv('year_predictions',header = FALSE)
for(i in 1:nrow(predictions)){ 
  if(i%%500 == 0){
    print(i)}
  a <- ye2[[i]]
  a_train <- forecast:::subset.ts(a,end=length(a)-hory)
  a_test <- forecast:::subset.ts(a, start = length(a)-hory+1)
  pred <-as.numeric(predictions[i,])
  mase <- ftsa:::error(forecast=pred, true=a_test,insampletrue=a_train, method='mase')
  mdase <- ftsa:::error(forecast=pred, true=a_test,insampletrue=a_train, method='mdase')
  smape <- ftsa:::error(forecast=pred, true=a_test,insampletrue=a_train, method='smape')
  rmsse <- ftsa:::error(forecast=pred, true=a_test,insampletrue=a_train, method='rmsse')
  smdape <- ftsa:::error(forecast=pred, true=a_test,insampletrue=a_train, method='smdape')
  
  list_of_metr[nrow(list_of_metr)+1,] <- c(mase,mdase,smape,rmsse,smdape)
}

meth_inf <- c(unique(as.numeric(which(!is.finite(as.matrix(list_of_metr)), arr.ind=TRUE)[,1])))
list_of_metr[meth_inf,] <- NA
colMeans(list_of_metr, na.rm=TRUE)

##### QUARTERLY DATA
##--------------------------------------------------------------------------------------    

list_of_metr <- data.frame(mase=double(),mdase=double(),smape=double(),rmsse=double(),smdape=double(),stringsAsFactors=FALSE) 
colnames(list_of_metr) <- c('mase','mdase','smape','rmsse','smdape')  
##--------------------------------------------------------------------------------------   

seas_qu2 <- readRDS("seasonalpart_shorten_quarter_data") ## сезонная часть
qu2 <- readRDS("shorten_quarter_data_1999") ## истинные укороченные ряды

predictions <- read.csv('predictions',header = FALSE)  ###### MY PREDICTIONS
for(i in 1:nrow(predictions)){ 
  if(i%%300 == 0){
    print(i)}
  a <- qu2[[i]]
  a_train <- forecast:::subset.ts(a,end=length(a)-horq)
  a_test <- forecast:::subset.ts(a, start = length(a)-horq+1)
  spart <- seas_qu2[[i]]
  spart_train <- forecast:::subset.ts(spart,end=length(spart)-horq)
  spart_test <- forecast:::subset.ts(spart, start = length(spart)-horq+1)
  pred <-as.numeric(predictions[i,])
  mase <- ftsa:::error(forecast=pred+spart_test, true=a_test,insampletrue=a_train, method='mase')
  mdase <- ftsa:::error(forecast=pred+spart_test, true=a_test,insampletrue=a_train, method='mdase')
  smape <- ftsa:::error(forecast=pred+spart_test, true=a_test,insampletrue=a_train, method='smape')
  rmsse <- ftsa:::error(forecast=pred+spart_test, true=a_test,insampletrue=a_train, method='rmsse')
  smdape <- ftsa:::error(forecast=pred+spart_test, true=a_test,insampletrue=a_train, method='smdape')
  
  list_of_metr[nrow(list_of_metr)+1,] <- c(mase,mdase,smape,rmsse,smdape)
}

meth_inf <- c(unique(as.numeric(which(!is.finite(as.matrix(list_of_metr)), arr.ind=TRUE)[,1])))
length(meth_inf)
list_of_metr[meth_inf,] <- NA
colMeans(list_of_metr, na.rm=TRUE)




##### MONTHLY DATA
##--------------------------------------------------------------------------------------    

list_of_metr <- data.frame(mase=double(),mdase=double(),smape=double(),rmsse=double(),smdape=double(),stringsAsFactors=FALSE) 
colnames(list_of_metr) <- c('mase','mdase','smape','rmsse','smdape')  
##--------------------------------------------------------------------------------------   

seas_mon2 <- readRDS("seasonalpart_shorten_month_data") ## сезонная часть
mon2 <- readRDS("shorten_month_data_2000") ## истинные укороченные ряды

predictions <- read.csv('month_holts_preds.csv',header = FALSE)  ###### MY PREDICTIONS
for(i in 1:nrow(predictions)){ #length(qu2)
  if(i%%300 == 0){
    print(i)}
  a <- mon2[[i]]
  a_train <- forecast:::subset.ts(a,end=length(a)-horm)
  a_test <- forecast:::subset.ts(a, start = length(a)-horm+1)
  spart <- seas_mon2[[i]]
  spart_train <- forecast:::subset.ts(spart,end=length(spart)-horm)
  spart_test <- forecast:::subset.ts(spart, start = length(spart)-horm+1)
  pred <-as.numeric(predictions[i,])
  mase <- ftsa:::error(forecast=pred*spart_test, true=a_test,insampletrue=a_train, method='mase')
  mdase <- ftsa:::error(forecast=pred*spart_test, true=a_test,insampletrue=a_train, method='mdase')
  smape <- ftsa:::error(forecast=pred*spart_test, true=a_test,insampletrue=a_train, method='smape')
  rmsse <- ftsa:::error(forecast=pred*spart_test, true=a_test,insampletrue=a_train, method='rmsse')
  smdape <- ftsa:::error(forecast=pred*spart_test, true=a_test,insampletrue=a_train, method='smdape')
  
  list_of_metr[nrow(list_of_metr)+1,] <- c(mase,mdase,smape,rmsse,smdape)
}

meth_inf <- c(unique(as.numeric(which(!is.finite(as.matrix(list_of_metr)), arr.ind=TRUE)[,1])))
length(meth_inf)
list_of_metr[meth_inf,] <- NA
colMeans(list_of_metr, na.rm=TRUE)










