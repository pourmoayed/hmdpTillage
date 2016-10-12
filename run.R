
library(hmdpTillage)

# Set HMDP parameters:
param<-setParam()

# Build the HMDP model
prefix<-"hmdp_"
BuildHMDP(prefix, param)

# Solve the HMDP model
wLbl<-"Reward"
durLbl<-"Time"
mdp<-loadMDP(prefix)
g<-policyIteAve(mdp,wLbl,durLbl)      # Finds the optimal policy using the average reward per week (g) criterion
policy<-getPolicy(mdp) # optimal policy for each sId

do.call(file.remove,list(list.files(pattern = prefix)))
rm(mdp)


size<-dim(param$disAvgWat)[1]*dim(param$disSdWat)[1]*dim(param$disSdPos)[1]*dim(param$disMeanPos)[1]*dim(param$disTem)[1]*dim(param$disPre)[1]
size*max(param$opD)*param$opNum*param$tMax

