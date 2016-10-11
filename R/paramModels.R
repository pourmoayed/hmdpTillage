
#' Set the parameters used when build the HMDP model
#'
#' @param opNum number of operations for scheduling
#' @param tMax Maximum length of growing cycle.
#' @param opSeq A vector showing operations indexes and sequence
#' @param opE A vector showing earliest start time of operations
#' @param opL A vector showing latest start time of operationss
#' @param opD A vector showing the time needed to complete operation (obtained by machine capacity and the field area)
#' @param opDelay A vector including the delay times after finishing  the operations (except the last operation)
#' @param opFixCost A vector including the fixed costs of tillage operations
#' @param watTh A vector including threshold values used as the optimal level of soil-water content for performing tillage operations (%)
#' @param coefLoss Coeficiant showing yield reduction when soil-water content is not appropriate
#' @param priceYield Price per kg yield (DDK)
#' @param machCap A vector showing the machine capacity for tillage operations
#' @param yieldHa Estimated yield per hetar in field (kg)
#' @param fieldArea Area of field (ha)
#' @param coefTimeliness Timeliness Coeficiant showing yield reduction resulting from postponing a tillage operation.
#' @param costSkip Cost of skipping tillage operations for the current cropping period
#' @param centerPointsAvgWat Center points for discritization of estimated mean of soil-water content
#' @param centerPointsSdWat Center points for discritization of estimated standard deviation of soil-water content
#' @param centerPointsSdPos Center points for discritization of estimated standard deviation of posterior distribution in SSM
#' @param centerPointsMeanPos Center points for discritization of estimated mean of posterior distribution in SSM
#' @param centerPointsTem Center points for discritization of weather forecast data regarding air temperature
#' @param centerPointsPre Center points for discritization of weather forecast data regarding precipitation
#' @param temMeanDry Mean of air temprature in dry days (\url{http://link.springer.com/article/10.1007/BF00142466}).
#' @param temMeanWet Mean of air temprature in wet days (\url{http://link.springer.com/article/10.1007/BF00142466}).
#' @param temVarDry variance of air temprature in dry days (\url{http://link.springer.com/article/10.1007/BF00142466}).
#' @param temVarWet variance of air temprature in wet days (\url{http://link.springer.com/article/10.1007/BF00142466}).
#' @param dryDayTh Threshold of precipitation amount for being a dry day (\url{http://link.springer.com/article/10.1007/BF00142466}).
#' @param precShape Shape parameter of gamma distribution for precipitation amount (\url{http://link.springer.com/article/10.1007/BF00142466}).
#' @param precScale Scale parameter of gamma distribution for precipitation amount (\url{http://link.springer.com/article/10.1007/BF00142466}).
#' @param prDryWet Probability of a wet day when the previous day is dry (\url{http://link.springer.com/article/10.1007/BF00142466}).
#' @param prWetWet Probability of a wet day when the previous day is wet (\url{http://link.springer.com/article/10.1007/BF00142466}).
#' @param hydroWatR Residual soil moisture value (\url{http://onlinelibrary.wiley.com/doi/10.1002/hyp.6629/abstract}).
#' @param hydroWatS Soil moisture value at saturation (\url{http://onlinelibrary.wiley.com/doi/10.1002/hyp.6629/abstract}).
#' @param hydroM Parameter used in the infiltration process of rainfall-runoff model (\url{http://onlinelibrary.wiley.com/doi/10.1002/hyp.6629/abstract}).
#' @param hydroKs Saturated hydralic conductivity used for drainage process in the rainfall-runoff model (\url{http://onlinelibrary.wiley.com/doi/10.1002/hyp.6629/abstract}).
#' @param hydroLamba Parameter used for drainage process in therainfall-runoff model (\url{http://onlinelibrary.wiley.com/doi/10.1002/hyp.6629/abstract}).
#' @param hydroETa Parameter used for evapotranpiration process in the rainfall-runoff model (\url{http://onlinelibrary.wiley.com/doi/10.1002/hyp.6629/abstract}).
#' @param hydroETb Parameter used for evapotranpiration process in the rainfall-runoff model (\url{http://onlinelibrary.wiley.com/doi/10.1002/hyp.6629/abstract}).
#' @param hydroETx Parameter used for evapotranpiration process in the rainfall-runoff model (\url{http://onlinelibrary.wiley.com/doi/10.1002/hyp.6629/abstract}).
#' @param gSSMW System variance of Gaussian SSM.
#' @param gSSMV Observation variance of Gaussian SSM.
#' @param gSSMm0 Initial posterior mean of Gaussian SSM.
#' @param gSSMc0 Initial posterior variance of Gaussian SSM.
#' @param nGSSMm0 Initial posterior mean of non-Gaussian SSM.
#' @param nGSSMc0 Initial posterior mean of non-Gaussian SSM.
#' @param nGSSMK Number of observations in non-Gaussian SSM.
#' @param check Check model e.g. do trans pr sum to one
#'
#' @return A list containing all the parameters used in three-level HMDP
#' @author Reza Pourmoayed \email{rpourmoayed@@econ.au.dk}
#' @export
setParam<-function(
  opNum=4,
  tMax=12,
  opSeq=c(1,2,3,4),
  opE=c(1,3,5,7),
  #opL=c(13,18,21,24),
  opL=c(6,8,10,12),
  opD=c(2,2,2,2),
  opDelay=c(0,0,0),
  opFixCost=c(1000,1000,1000,1000),
  watTh=c(30,40,40,50),
  coefLoss=0.3,
  priceYield=10,
  machCap=50,
  yieldHa=100,
  fieldArea=100,
  coefTimeliness=0.01,
  costSkip= 80000,
  # centerPointsAvgWat=seq(hydroWatR+1,hydroWatS, by=5),#seq(5,60, by=6),
  # centerPointsSdWat=seq(1,8, by=2),
  # centerPointsSdPos=seq(0.1,0.8, by=0.3),
  # centerPointsMeanPos=seq(0.5,1.5,by=0.1),
  # centerPointsTem=seq(15,25,by=2),
  # centerPointsPre=c(0,dryDayTh*2, seq(1,7,by=2) ),

  centerPointsAvgWat=seq(hydroWatR+1,hydroWatS, by=15),#seq(5,60, by=6),
  centerPointsSdWat=seq(1,3, by=2),
  centerPointsSdPos=seq(0.1,0.4, by=0.3),
  centerPointsMeanPos=seq(0.7,1.2,by=0.3),
  centerPointsTem=seq(15,20,by=2),
  centerPointsPre= c(0,dryDayTh*2, seq(1.5,8,by=2) ),


  temMeanDry=19, #shoule be estimates
  temMeanWet=12, #shoule be estimates
  temVarDry=16, #shoule be estimates
  temVarWet=16, #shoule be estimates
  dryDayTh=0.25, #shoule be estimates
  precShape=0.676, #shoule be estimates
  precScale=0.744, #shoule be estimates
  prDryWet=0.175,  #shoule be estimates
  prWetWet=0.480, #shoule be estimates

  hydroWatR=7.5, #shoule be estimates
  hydroWatS=57, #shoule be estimates
  hydroM=15, #shoule be estimates
  hydroKs=23,  #shoule be estimates
  hydroLamba=0.423, #shoule be estimates
  hydroETa=-2, #shoule be estimates
  hydroETb=1.26, #shoule be estimates
  hydroETx=0.6, #shoule be estimates

  gSSMW=0.01,
  gSSMV=25,
  gSSMm0=1,
  gSSMc0=0.2,
  nGSSMm0=4,
  nGSSMc0=2,
  nGSSMK=20,

  check = FALSE
){
   model<-list(opNum=opNum)
   model$opSeq<-opSeq
   model$tMax<-tMax
   model$opE<-opE
   model$opL<-opL
   model$opD<-opD
   model$opDelay<-opDelay
   model$opFixCost<-opFixCost
   model$watTh<-watTh
   model$coefLoss<-coefLoss
   model$priceYield<-priceYield
   model$machCap<-machCap
   model$yieldHa<-yieldHa
   model$fieldArea<-fieldArea
   model$coefTimeliness<-coefTimeliness
   model$costSkip<-costSkip
   model$temMeanDry<-temMeanDry
   model$temMeanWet<-temMeanWet
   model$temVarDry<-temVarDry
   model$temVarWet<-temVarWet
   model$dryDayTh<-dryDayTh
   model$precShape<-precShape
   model$precScale<-precScale
   model$prDryWet<-prDryWet
   model$prWetWet<-prWetWet

   model$hydroWatR<-hydroWatR
   model$hydroWatS<-hydroWatS
   model$hydroM<-hydroM
   model$hydroKs<-hydroKs
   model$hydroLamba<-hydroLamba
   model$hydroETa<-hydroETa
   model$hydroETb<-hydroETb
   model$hydroETx<-hydroETx

   model$gSSMW<-gSSMW
   model$gSSMV<-gSSMV
   model$gSSMm0<-gSSMm0
   model$gSSMc0<-gSSMc0
   model$nGSSMm0<-nGSSMm0
   model$nGSSMc0<-nGSSMc0
   model$nGSSMK<-nGSSMK

   model$check <- check

   model$centerPointsAvgWat<-centerPointsAvgWat
   model$centerPointsSdWat<-centerPointsSdWat
   model$centerPointsSdPos<-centerPointsSdPos
   model$centerPointsMeanPos<-centerPointsMeanPos
   model$centerPointsTem<-centerPointsTem
   model$centerPointsPre<-centerPointsPre

   #Discritization of continious states:

   obj<-Discretize()
   disAvgWat<-matrix()
   disSdWat<-matrix()
   disSdPos<-matrix()
   disMeanPos<-matrix()
   disTem<-matrix()
   disPre<-matrix()

   disAvgWat<-as.matrix(obj$discretize1DVec(centerPointsAvgWat, mInf=-1000, inf=1000, asDF=F), ncol=3)
   disSdWat<-as.matrix(obj$discretize1DVec(centerPointsSdWat, inf=100, mInf=0.01, asDF=F), ncol=3)
   disSdPos<-as.matrix(obj$discretize1DVec(centerPointsSdPos, inf=100, mInf=0, asDF=F), ncol=3)
   disMeanPos<-as.matrix(obj$discretize1DVec(centerPointsMeanPos, inf=100, asDF=F), ncol=3)
   disTem<-as.matrix(obj$discretize1DVec(centerPointsTem, inf=100, asDF=F), ncol=3)
   disPre<-as.matrix(obj$discretize1DVec(centerPointsPre, inf=100, mInf=0, asDF=F), ncol=3)

   model$disAvgWat<-disAvgWat
   model$disSdWat<-disSdWat
   model$disSdPos<-disSdPos
   model$disMeanPos<-disMeanPos
   model$disTem<-disTem
   model$disPre<-disPre

   return(model)
}
