#include "hmdp.h"

// ===================================================

const double HMDP::ZERO = 4.656613e-10;   // equal 1/2147483647 (limit of int since trans pr is stored as int when build the hgf)

// ===================================================

HMDP::HMDP(const string prefix, const List paramModel): w(prefix) {

   List rParam(paramModel);       // Get parameters in params
   opNum=as<int>(rParam["opNum"]);
   tMax=as<int>(rParam["tMax"]);
   opSeq=as<arma::vec>(rParam["opSeq"]);
   opE=as<arma::vec>(rParam["opE"]);
   opL=as<arma::vec>(rParam["opL"]);
   opD=as<arma::vec>(rParam["opD"]);
   opDelay=as<arma::vec>(rParam["opDelay"]);
   opFixCost=as<arma::vec>(rParam["opFixCost"]);
   watTh=as<arma::vec>(rParam["watTh"]);
   coefLoss=as<double>(rParam["coefLoss"]);
   priceYield=as<double>(rParam["priceYield"]);
   machCap=as<double>(rParam["machCap"]);
   yieldHa=as<double>(rParam["yieldHa"]);
   fieldArea=as<double>(rParam["fieldArea"]);
   coefTimeliness=as<double>(rParam["coefTimeliness"]);
   costSkip=as<double>(rParam["costSkip"]);

   temMeanDry=as<double>(rParam["temMeanDry"]);
   temMeanWet=as<double>(rParam["temMeanWet"]);
   temVarDry=as<double>(rParam["temVarDry"]);
   temVarWet=as<double>(rParam["temVarWet"]);
   dryDayTh=as<double>(rParam["dryDayTh"]);
   precShape=as<double>(rParam["precShape"]);
   precScale=as<double>(rParam["precScale"]);
   prDryWet=as<double>(rParam["prDryWet"]);
   prWetWet=as<double>(rParam["prWetWet"]);

   hydroWatR=as<double>(rParam["hydroWatR"]);
   hydroWatS=as<double>(rParam["hydroWatS"]);
   hydroM=as<double>(rParam["hydroM"]);
   hydroKs=as<double>(rParam["hydroKs"]);
   hydroLamba=as<double>(rParam["hydroLamba"]);
   hydroETa=as<double>(rParam["hydroETa"]);
   hydroETb=as<double>(rParam["hydroETb"]);
   hydroETx=as<double>(rParam["hydroETx"]);

   gSSMW=as<double>(rParam["gSSMW"]);
   gSSMV=as<double>(rParam["gSSMV"]);
   gSSMm0=as<double>(rParam["gSSMm0"]);
   gSSMc0=as<double>(rParam["gSSMc0"]);
   nGSSMm0=as<double>(rParam["nGSSMm0"]);
   nGSSMc0=as<double>(rParam["nGSSMc0"]);
   nGSSMK=as<double>(rParam["nGSSMK"]);

   check = as<bool>(rParam["check"]);

   dMP = as<arma::mat>(rParam["disMeanPos"]);
   dSP = as<arma::mat>(rParam["disSdPos"]);
   dMW = as<arma::mat>(rParam["disAvgWat"]);
   dSW = as<arma::mat>(rParam["disSdWat"]);
   dT = as<arma::mat>(rParam["disTem"]);
   dP = as<arma::mat>(rParam["disPre"]);

   sMP = as<arma::vec>(rParam["centerPointsMeanPos"]);
   sSP = as<arma::vec>(rParam["centerPointsSdPos"]);
   sMW = as<arma::vec>(rParam["centerPointsAvgWat"]);
   sSW = as<arma::vec>(rParam["centerPointsSdWat"]);
   sT = as<arma::vec>(rParam["centerPointsTem"]);
   sP = as<arma::vec>(rParam["centerPointsPre"]);

   sizeSMP = sMP.size();
   sizeSSP = sSP.size();
   sizeSMW = sMW.size();
   sizeSSW = sSW.size();
   sizeST = sT.size();
   sizeSP = sP.size();

   // matrices for filling the rewards and transition probabilities before running the HMDP:
   prMW = vector <vector<vector< vector< vector< vector<double> > > > > >(sizeSMW,
          vector<vector< vector< vector< vector<double> > > > >(sizeSMP,
          vector<vector< vector< vector<double> > > >(sizeSSP,
          vector<vector< vector<double> > >(sizeST,
          vector<vector<double> > (sizeSP,
          vector <double>(sizeSMW) ) ) ) ) ); //prMW[iMWt][iMPt][iSPt][iTt][iPt][iMW]

   prMP = vector< vector< vector<double> > > (sizeSMP,
          vector< vector<double> >(sizeSSP,
          vector<double>(sizeSMP) ) ); //prMP[iMPt][iSPt][iMP]

   prSP = vector<vector< vector< vector< vector<double> > > > >(sizeSMW,
          vector<vector< vector< vector<double> > > >(sizeSSP,
          vector<vector< vector<double> > >(sizeST,
          vector<vector<double> > (sizeSP,
          vector <double>(sizeSSP) ) ) ) ); // prSP[iMWt][iSPt][iTt][iPt][iSP]


   prSW = vector < vector< vector<double> > > (tMax+1,
          vector< vector<double> >(sizeSSW,
          vector<double>(sizeSSW) ) ); //prSW[t][iSWt][iSW]

   prT =  vector <vector< vector<double> > > (sizeST,
          vector< vector<double> > (sizeSP,
          vector<double>(sizeST) ) ); //prT[iTt][iPt][iT]

   prP = vector< vector<double> > (sizeSP,
          vector<double>(sizeSP) ); //prP[iPt][iP]

   rewDo = vector <vector< vector<double> > >(opNum,
           vector< vector<double> >(sizeSMW,
           vector<double>(sizeSSW) ) ); //rewDo[op][iMWt][iSWt]


   int opDMax = arma::max(opD);
   int opSMAX = tMax; //arma::max(opL-opE);


   mapL2Vector = vector< vector< vector< vector< vector< vector< vector< vector<int> > > > > > > >(opSMAX+1,
                 vector< vector< vector< vector< vector< vector< vector<int> > > > > > >(opDMax+1,
                 vector< vector< vector< vector< vector< vector<int> > > > > >(sizeSMW,
                 vector< vector< vector< vector< vector<int> > > > >(sizeSSW,
                 vector< vector< vector< vector<int> > > >(sizeSMP,
                 vector< vector< vector<int> > >(sizeSSP,
                 vector< vector<int> >(sizeST,
                 vector <int>(sizeSP) ) ) ) ) ) ) ) ; //mapL2Vector[opS][d][iMW][iSW][iMP][iSP][iT][iP];

   mapLVector =  vector< vector< vector< vector< vector< vector< vector<int> > > > > > >(opSMAX+1,
                 vector< vector< vector< vector< vector< vector<int> > > > > >(sizeSMW,
                 vector< vector< vector< vector< vector<int> > > > >(sizeSSW,
                 vector< vector< vector< vector<int> > > >(sizeSMP,
                 vector< vector< vector<int> > >(sizeSSP,
                 vector< vector<int> >(sizeST,
                 vector <int>(sizeSP) ) ) ) ) ) ) ; //mapLVector[opS][iMW][iSW][iMP][iSP][iT][iP];
}

// ===================================================

void HMDP::Preprocess() {
   Rcout << "Build the HMDP ... \n\nStart preprocessing ...\n"<<endl;
   CalcTransPrMW();
   CalcTransPrSW();
   CalcTransPrMP();
   CalcTransPrSP();
   CalcTransPrT();
   CalcTransPrP();
   CalcRewaerdDo();
   Rcout << "... finished preprocessing.\n";
}

// ===================================================

SEXP HMDP::BuildHMDP() {
   Preprocess();

   Rcout << "Start writing to binary files ... \n";
   w.SetWeight("Time");
   w.SetWeight("Reward");

   w.Process();    // level 0 (founder)
   w.Stage();
   w.State("Dummy");
//   BuildMapL1Vector(0);
   WeightTransPrIni();  //calculate the initial transition probabilities
   w.Action(scope, index, pr, weights, "Dummy", false);
      BuildL1Process();
   w.EndAction();
   w.EndState();
   w.EndStage();
   w.EndProcess();  // end level 1 (founder)
   w.CloseWriter();
   Rcout << "... finished writing to binary files.\n";
   return wrap(w.log.str());
}

// ===================================================


void HMDP::BuildL1Process() {
   w.Process(); // level 1
      BuildL1Operation();
      BuildL1StageLast();
   w.EndProcess(); // end level
}


// ===================================================


void HMDP::BuildL1Operation() {
   int iMW, iSW, iMP, iSP, iT, iP, opS, op, d;
      for(op=0; op<opNum; op++){
         if ( op<(opNum-1) ) BuildMapL1Vector(op+1);
         w.Stage();
            // Dummy state to define the subprocesses
            w.State("Dummy");
            scope.assign(1,2); index.assign(1,0); pr.assign(1,1);
            weights.assign(2,0);
            label = "Op" + ToString<int>(op);
            w.Action(scope, index, pr, weights, label, false);
            BuildL2Process(op);
            w.EndAction();
            w.EndState();

            for(opS=opE[op]; opS<=opL[op]; opS++){
              if( (op==0) & (opS!=1) ) continue;
              if(opS>opL[op]-opD[op]) continue;
              for(iMW=0; iMW<sizeSMW; iMW++){
                for(iSW=0; iSW<sizeSSW; iSW++){
                  for(iMP=0; iMP<sizeSMP; iMP++){
                    for(iSP=0; iSP<sizeSSP; iSP++){
                      for(iT=0; iT<sizeST; iT++){
                        for(iP=0; iP<sizeSP; iP++){
                          label = getLabel(op,opS,iMW,iSW,iMP,iSP,iT,iP);
                          w.State(label);
                          d=opD[op];
                          index.assign(1,mapR[getLabel(op,opS,d,iMW,iSW,iMP,iSP,iT,iP)]);
                          scope.assign(1,3);
                          pr.assign(1,1);
                          weights.assign(2,0);
                          weights[1]=-opFixCost[op];
                          label = "new op = " + ToString<int>(op);
                          w.Action(scope, index, pr, weights, label, true);
                          w.EndState();
                        }
                      }
                    }
                  }
                }
              }
            }
            w.State("dummy");
            scope.assign(1,0); index.assign(1,0); pr.assign(1,1); weights.assign(2,0);
            w.Action(scope, index, pr, weights, "skiped", true);
            w.EndState();
         w.EndStage();
      }
 }


// ===================================================


void HMDP::BuildL1StageLast() {
   w.Stage();
      w.State("dummy");
         scope.assign(1,0); index.assign(1,0); pr.assign(1,1);
         weights.assign(2,0);
         w.Action(scope, index, pr, weights, "finished", true);
      w.EndState();
   w.EndStage();
}

// ===================================================
void HMDP::BuildL2Process(int op) {
   int t, iMW, iSW, iMP, iSP, iT, iP, opS, d;
   w.Process(); // level 2
      for(t=opE[op]; t<=opL[op]; t++) {
         if(t != opL[op]) BuildMapL2Vector(t+1,op);
         w.Stage();
             for(opS=opE[op];opS<=t;opS++){
               if( (op==0) & (opS!=1) ) continue;
               if(opS>opL[op]-opD[op]) continue;
               for(d=0; d<=opD[op]; d++){
                 if( opD[op]-t+opS>d) continue;
                 for(iMW=0; iMW<sizeSMW; iMW++){
                   for(iSW=0; iSW<sizeSSW; iSW++){
                     for(iMP=0; iMP<sizeSMP; iMP++){
                       for(iSP=0; iSP<sizeSSP; iSP++){
                         for(iT=0; iT<sizeST; iT++){
                           for(iP=0; iP<sizeSP; iP++){
                             label = getLabel(op,opS,d,iMW,iSW,iMP,iSP,iT,iP,t);
                             idS = w.State(label);
                             if (t==opS) mapR[getLabel(op,opS,d,iMW,iSW,iMP,iSP,iT,iP)] = idS;
                             if ( (d>0) & (d<=opL[op]-t) ){
                               WeightsTransPos(op,opS,d,iMW,iSW,iMP,iSP,iT,iP,t);
                               w.Action(scope, index, pr, weights, "pos.", true);
                               WeightsTransDo(op,opS,d,iMW,iSW,iMP,iSP,iT,iP,t);
                               w.Action(scope, index, pr, weights, "do.", true);
                             }
                             if(d==0){
                               WeightsTransTerm(op,opS,d,iMW,iSW,iMP,iSP,iT,iP,t);
                               w.Action(scope, index, pr, weights, "term.", true);
                             }
                             if(d>opL[op]-t){
                               WeightsTransSkip(op);
                               w.Action(scope, index, pr, weights, "skip.", true);
                             }
                             w.EndState();
                           }
                         }
                       }
                     }
                   }
                 }
               }
             }
         w.EndStage();
      }
   w.EndProcess(); // end level
}


// ===================================================

void HMDP::WeightsTransPos(int & op, int & opSt, int & dt, int & iMWt, int & iSWt, int & iMPt, int & iSPt, int & iTt, int & iPt, int & t) {
   double pr4, prS;
   int opS,d,iMW,iSW,iMP,iSP,iT,iP, id;
   opS=opSt;
   d=dt;

   pr.clear(); index.clear(); scope.clear();
   for(iMW=0; iMW<sizeSMW; iMW++){
     for(iSW=0; iSW<sizeSSW; iSW++){
       for(iMP=0; iMP<sizeSMP; iMP++){
         for(iSP=0; iSP<sizeSSP; iSP++){
           for(iT=0; iT<sizeST; iT++){
             for(iP=0; iP<sizeSP; iP++){
               id=mapL2Vector[opS][d][iMW][iSW][iMP][iSP][iT][iP];
               if(id<0) Rcout<<"error"<<endl;
               prS = prSP[iMWt][iSPt][iTt][iPt][iSP];
               pr4 = prS*exp(prMW[iMWt][iMPt][iSPt][iTt][iPt][iMW] + prSW[t][iSWt][iSW] + prMP[iMPt][iSPt][iMP]
                           + prT[iTt][iPt][iT] + prP[iPt][iP]);
               if (pr4>ZERO) {
                 pr.push_back(pr4); index.push_back(id);
               }
             }
           }
         }
       }
     }
   }
   scope.assign(pr.size(),1);
   weights.assign(2,0);
   weights[0]=1;
   weights[1]=-coefTimeliness*priceYield*yieldHa*fieldArea;
      if (check) {
      arma::vec tmp(pr);
      if (!Equal(sum(tmp),1,1e-8)) {
         Rcout << "Warning sum pr!=1 in WeightsTransPrPos - diff = " << 1-sum(tmp) << " op = " << op << " action = pos. " << " index:" << endl; //vec2String<int>(index) << " pr:" << vec2String<flt>(pr) << endl;
      }
   }
}

// ===================================================

void HMDP::WeightsTransDo(int & op, int & opSt, int & dt, int & iMWt, int & iSWt, int & iMPt, int & iSPt, int & iTt, int & iPt, int & t) {
  double pr4, prS;
  int opS,d,iMW,iSW,iMP,iSP,iT,iP, id;
  string str;
  opS=opSt;
  d=dt-1;

  pr.clear(); index.clear(); scope.clear();
  for(iMW=0; iMW<sizeSMW; iMW++){
    for(iSW=0; iSW<sizeSSW; iSW++){
      for(iMP=0; iMP<sizeSMP; iMP++){
        for(iSP=0; iSP<sizeSSP; iSP++){
          for(iT=0; iT<sizeST; iT++){
            for(iP=0; iP<sizeSP; iP++){
              id=mapL2Vector[opS][d][iMW][iSW][iMP][iSP][iT][iP];
              if(id<0) Rcout<<"error"<<endl;
              prS = prSP[iMWt][iSPt][iTt][iPt][iSP];
              pr4 = prS*exp(prMW[iMWt][iMPt][iSPt][iTt][iPt][iMW] + prSW[t][iSWt][iSW] + prMP[iMPt][iSPt][iMP]
                              + prT[iTt][iPt][iT] + prP[iPt][iP]);
              if (pr4>ZERO) {
                pr.push_back(pr4); index.push_back(id);
              }
            }
          }
        }
      }
    }
  }
  scope.assign(pr.size(),1);
  weights[0]=1;
  weights[1]=rewDo[op][iMWt][iSWt];
  if (check) {
    arma::vec tmp(pr);
    if (!Equal(sum(tmp),1,1e-8)) {
      Rcout << "Warning sum pr!=1 in WeightsTransPrDo - diff = " << 1-sum(tmp) << " op = " << op << " action = Do. " << endl; // " index:" << vec2String<int>(index) << " pr:" << vec2String<flt>(pr) << endl;
    }
  }
}

// ===================================================

void HMDP::WeightsTransTerm(int & op, int & opSt, int & dt, int & iMWt, int & iSWt, int & iMPt, int & iSPt, int & iTt, int & iPt, int & t) {
  int id,opS;
  opS=t;

  pr.clear(); index.clear(); scope.clear();
  if(op!=(opNum-1)){
    id=mapLVector[opS][iMWt][iSWt][iMPt][iSPt][iTt][iPt];
    if(id<0) Rcout<<"error"<<endl;
    scope.assign(1,0);
    index.assign(1, id);
    pr.assign(1,1);
    weights.assign(2,0);
  }
  else{
    id=0;
    scope.assign(1,0);
    index.assign(1, id);
    pr.assign(1,1);
    weights.assign(2,0); //here what is the length to the next copping cycle??????
  }
}

// ===================================================

void HMDP::WeightsTransSkip(int & op){
  int id;

  pr.clear(); index.clear(); scope.clear();
  if(op!=(opNum-1)){
    id=mapL1["dummy"];
  }else{
    id=0;
  }
  scope.assign(1,0);
  index.assign(1, id);
  pr.assign(1,1);
  weights.assign(2,0); //here what is the length to the next copping cycle??????
  weights[1]=-costSkip;
}

// ===================================================

void HMDP::WeightTransPrIni(){
  pr.clear(); index.clear(); scope.clear();
  //int iMWt, iMW, iSW, iMP, iSP, iT, iP, opS, id;
  //double pr4;
  // opS = 1;
  // iMWt = 0;
  // iMP = 0; //we need to define it???????????
  // iSP = 0; //we need to define it???????????
  // iSW = 0; //we need to define it???????????
  // for(iMW=0; iMW<sizeSMW; iMW++){
  //   for(iT=0; iT<sizeST; iT++){
  //     for(iP=0; iP<sizeSP; iP++){
  //             id=mapLVector[opS][iMW][iSW][iMP][iSP][iT][iP];
  //             pr4 = exp(prMW[iMWt][iMP][iSP][iT][iP][iMW]
  //                              + prT[iTt][iPt][iT] + prP[0][iP]);
  //             if (pr4>ZERO) {
  //               pr.push_back(pr4); index.push_back(id);
  //             }
  //           }
  //         }
  //       }
  //scope.assign(pr.size(),2);
  index.assign(1,1);
  scope.assign(1,2);
  pr.assign(1,1);
  weights.assign(2,0); weights[1]=priceYield*yieldHa*fieldArea;
  }

// ===================================================

void HMDP::BuildMapL1Vector(int op) {
   int iMW, iSW, iMP, iSP, iT, iP, opS, idL1;
   // level 1
   idL1=1;
   mapL1.clear();

   for(opS=opE[op]; opS<=opL[op]; opS++){
     if( (op==0) & (opS!=1) ) continue;
     if(opS>opL[op]-opD[op]) continue;
     for(iMW=0; iMW<sizeSMW; iMW++){
       for(iSW=0; iSW<sizeSSW; iSW++){
         for(iMP=0; iMP<sizeSMP; iMP++){
           for(iSP=0; iSP<sizeSSP; iSP++){
             for(iT=0; iT<sizeST; iT++){
               for(iP=0; iP<sizeSP; iP++){
                 mapLVector[opS][iMW][iSW][iMP][iSP][iT][iP] =-1;
                 mapLVector[opS][iMW][iSW][iMP][iSP][iT][iP] =idL1;
                 idL1++;
               }
             }
           }
         }
       }
     }
   }
   mapL1["dummy"] = idL1;
}


// ===================================================

void HMDP::BuildMapL2Vector(int t, int op) {
  int iMW, iSW, iMP, iSP, iT, iP, opS, d, idL2;
   // level 2
   idL2=0;

   for(opS=opE[op];opS<=t;opS++){
     if( (op==0) & (opS!=1) ) continue;
     if(opS>opL[op]-opD[op]) continue;
     for(d=0; d<=opD[op]; d++){
       if( opD[op]-t+opS>d ) continue;
       for(iMW=0; iMW<sizeSMW; iMW++){
         for(iSW=0; iSW<sizeSSW; iSW++){
           for(iMP=0; iMP<sizeSMP; iMP++){
             for(iSP=0; iSP<sizeSSP; iSP++){
               for(iT=0; iT<sizeST; iT++){
                 for(iP=0; iP<sizeSP; iP++){
                   mapL2Vector[opS][d][iMW][iSW][iMP][iSP][iT][iP]=-1;
                   mapL2Vector[opS][d][iMW][iSW][iMP][iSP][iT][iP]=idL2;
                   idL2++;
                   }
                 }
               }
             }
           }
         }
       }
     }
   }

// ===================================================

void HMDP::CalcRewaerdDo(){
  cpuTime.Reset(0); cpuTime.StartTime(0);
  int iMW,iSW,op;
  for(op=0; op<opNum; op++){
    for(iMW=0;iMW<sizeSMW;iMW++){
      for(iSW=0;iSW<sizeSSW;iSW++){
        rewDo[op][iMW][iSW]= -coefLoss*priceYield*yieldHa*machCap*(1- R::pnorm(watTh[op],dMW(iMW,0),dSW(iSW,0),1,0) );
      }
    }
  }
}

// ===================================================
void HMDP::CalcTransPrMW(){   //prMW[iMWt][iMPt][iSPt][iTt][iPt][iMW]
  cpuTime.Reset(0); cpuTime.StartTime(0);
  int iMWt, iMPt,iSPt,iTt,iPt,iMW;
  double lower, upper, mt, ct, ft;

  for(iMWt=0; iMWt<sizeSMW; iMWt++){
    for(iMPt=0;iMPt<sizeSMP;iMPt++){
      for(iSPt=0;iSPt<sizeSSP;iSPt++){
        for(iTt=0;iTt<sizeST;iTt++){
          for(iPt=0;iPt<sizeSP;iPt++){
            ft = Hydro(dMW(iMWt,0),dT(iTt,0),dP(iPt,0));
            mt=ft*dMP(iMPt,0); ct= pow(ft,2)*( pow(dSP(iSPt,0),2) + gSSMW ) + gSSMV;
            for(iMW=0; iMW<sizeSMW; iMW++){
              lower= dMW(iMW,1); upper=dMW(iMW,2);
              prMW[iMWt][iMPt][iSPt][iTt][iPt][iMW] = log( R::pnorm(upper,mt,sqrt(ct),1,0) - R::pnorm(lower,mt,sqrt(ct),1,0)  );
            }
          }
        }
      }
    }
  }
}

// ===================================================

void HMDP::CalcTransPrSW(){  //prSW[t][iSWt][iSW]
  cpuTime.Reset(0); cpuTime.StartTime(0);
  int t,iSWt,iSW;

  for(t=1; t<tMax; t++){
    for(iSWt=0; iSWt<sizeSSW; iSWt++){
      for(iSW=0; iSW<sizeSSW; iSW++){
        prSW[t][iSWt][iSW] = log(PrNGSSM( t, pow(dSW(iSW,1),2), pow(dSW(iSW,2),2), pow(dSW(iSWt,0),2) ) );
      }
    }
  }
}


// ===================================================

void HMDP::CalcTransPrMP(){ //prMP[iMPt][iSPt][iMP]
  cpuTime.Reset(0); cpuTime.StartTime(0);
  int iMPt,iSPt,iMP;
  double lower, upper, mt, ct;

  for(iMPt=0;iMPt<sizeSMP;iMPt++){
    for(iSPt=0;iSPt<sizeSSP;iSPt++){
      mt = dMP(iMPt,0); ct=pow(dSP(iSPt,0),2);
      for(iMP=0;iMP<sizeSMP;iMP++){
        lower= dMP(iMP,1); upper= dMP(iMP,2);
        prMP[iMPt][iSPt][iMP] = log( R::pnorm(upper,mt,sqrt(ct),1,0) - R::pnorm(lower,mt,sqrt(ct),1,0) );
      }
    }
  }
}

// ===================================================

void HMDP::CalcTransPrSP(){ //prSP[iMWt][iSPt][iTt][iPt][iSP]
  cpuTime.Reset(0); cpuTime.StartTime(0);
  int iMWt,iSPt,iTt,iPt,iSP;
  double ft, ct, cN;

  for(iMWt=0;iMWt<sizeSMW;iMWt++){
    for(iSPt=0;iSPt<sizeSSP;iSPt++){
      for(iTt=0;iTt<sizeST;iTt++){
        for(iPt=0;iPt<sizeSP;iPt++){
          ft=Hydro(dMW(iMWt,0),dT(iTt,0),dP(iPt,0));
          ct=pow(dSP(iSPt,0),2);
          cN= ( (ct+gSSMW)*gSSMV )/( pow(ft,2)*(ct+gSSMW) + gSSMV);
          for(iSP=0;iSP<sizeSSP;iSP++){
            if( ( cN>pow(dSP(iSP,1),2) ) & ( cN<=pow(dSP(iSP,2),2) ) ){
              prSP[iMWt][iSPt][iTt][iPt][iSP]=1;
            }else{
              prSP[iMWt][iSPt][iTt][iPt][iSP]=0;
            }
          }
        }
      }
    }
  }
}

// ===================================================

void HMDP::CalcTransPrT(){ //prT[iTt][iPt][iT]
  cpuTime.Reset(0); cpuTime.StartTime(0);
  int iTt,iT,iPt;
  double lower, upper, mt, ct;

  for(iTt=0; iTt<sizeST; iTt++){
    for(iPt=0; iPt<sizeSP; iPt++){
      if(dP(iPt,0)<=dryDayTh){
        mt=temMeanDry; ct=temVarDry;
      }else{
        mt=temMeanWet; ct=temVarWet;
      }
      for(iT=0; iT<sizeST; iT++){
       lower=dT(iT,1); upper=dT(iT,2);
       prT[iTt][iPt][iT] = log( R::pnorm(upper,mt,sqrt(ct),1,0) - R::pnorm(lower,mt,sqrt(ct),1,0) );
      }
    }
  }
}
// ===================================================

void HMDP::CalcTransPrP(){ //prP[iPt][iP]
  cpuTime.Reset(0); cpuTime.StartTime(0);
  int iPt,iP;
  double lower, upper; //, lowert, uppert;

  for(iPt=0; iPt<sizeSP; iPt++){
    //uppert=dP(iPt,1); lowert=dP(iPt,2);
    for(iP=0; iP<sizeSP; iP++){
      lower=dP(iP,1); upper=dP(iP,2);
      //if( (uppert<=dryDayTh) & (upper<=dryDayTh) ) prP[iPt][iP]= log(1-prDryWet);
      //if( (lowert>dryDayTh) & (upper<=dryDayTh) ) prP[iPt][iP]= log(1-prWetWet);
      //if( (uppert<=dryDayTh) & (lower>dryDayTh) ) prP[iPt][iP]= log(prDryWet*( R::pgamma(upper,precShape,precScale,1,0) - R::pgamma(lower,precShape,precScale,1,0)) );
      //if( (lowert>dryDayTh) & (lower>dryDayTh) ) prP[iPt][iP]= log(prWetWet*( R::pgamma(upper,precShape,precScale,1,0) - R::pgamma(lower,precShape,precScale,1,0) ) );

      // if( (dP(iPt,0)<=dryDayTh) & (dP(iP,0)<=dryDayTh) ) prP[iPt][iP]= log(1-prDryWet);
      // if( (dP(iPt,0)>dryDayTh) & (dP(iP,0)<=dryDayTh) ) prP[iPt][iP]= log(1-prWetWet);
      // if( (dP(iPt,0)<=dryDayTh) & (dP(iP,0)>dryDayTh) ) prP[iPt][iP]= log(prDryWet*( R::pgamma(upper,precShape,precScale,1,0) - R::pgamma(lower,precShape,precScale,1,0)) );
      // if( (dP(iPt,0)>dryDayTh) & (dP(iP,0)>dryDayTh) ) prP[iPt][iP]= log(prWetWet*( R::pgamma(upper,precShape,precScale,1,0) - R::pgamma(lower,precShape,precScale,1,0) ) );

      if( (dP(iPt,0)<=dryDayTh) & (dP(iP,0)<=dryDayTh) ) prP[iPt][iP]= log(1-prDryWet);
      if( (dP(iPt,0)>dryDayTh) & (dP(iP,0)<=dryDayTh) ) prP[iPt][iP]= log(1-prWetWet);
      if( (dP(iPt,0)<=dryDayTh) & (dP(iP,0)>dryDayTh)  ) { if(iP==1)lower=0; prP[iPt][iP]= log(prDryWet*( R::pgamma(upper,precShape,precScale,1,0) - R::pgamma(lower,precShape,precScale,1,0)) ); }
      if( (dP(iPt,0)>dryDayTh) & (dP(iP,0)>dryDayTh) ) { if(iP==1)lower=0; prP[iPt][iP]= log(prWetWet*( R::pgamma(upper,precShape,precScale,1,0) - R::pgamma(lower,precShape,precScale,1,0) ) ); }
    }
  }
}

// ===================================================

double HMDP::Hydro(double & Wt, double & Tt, double Pt){ // ssm.hydro(dMW(iMWt,0),dT(iTt,0),dP(iPt,0));

  double f, e, g, ET;

  ET= hydroETa + hydroETb*hydroETx*(0.46*Tt + 8.13);
  //f= hydroKs*( 1-(hydroFi*(hydroWatS-Wt)/hydroF) );
  f=Pt*(1- pow((Wt-hydroWatR)/(hydroWatS-hydroWatR),hydroM) );
  g= hydroKs*pow((Wt-hydroWatR)/(hydroWatS-hydroWatR),3+2/hydroLamba);
  e= ET*(Wt-hydroWatR)/(hydroWatS-hydroWatR);

  return(Wt+f-e-g);
}


// ===================================================
double HMDP::PrNGSSM(int t, double lower, double upper, double var) { //SOLVED[Reza] : Based on the formulatiom for this probability in the paper, I changed "n" to "nf" (nf is the sample size).

  double probSd, xUpper, xLower;
  double a, s, alpha, gamma, beta;
  double G=1;

  double oShape = (double) (nGSSMK-1)/(2);   //shape parameter of observation distribution
  double iShape = (double) (nGSSMK-1)/(2);//(double) (numSample-3)/(numSample-5); //("shape parameter of prior at t=1"): c_1 in the paper

  a = (double) (G * var * ( iShape + oShape*t ) ) / ( iShape + oShape*(t+1) ) ; // location
  s = a; // scale
  alpha = oShape; // shape 1
  gamma = iShape + oShape*t + 1; // shape 2
  beta =1 ;  // Weibul parameter
  xUpper = (double) (1)/( 1 + pow ( (double)(upper-a)/(s),-beta ) );
  xLower = (double) (1)/( 1 + pow ( (double)(lower-a)/(s),-beta ) );

  probSd= R::pbeta(xUpper, alpha, gamma,1, 0) - R::pbeta(xLower, alpha, gamma,1, 0);
  if( ( R::pbeta(xUpper, alpha, gamma,1, 0) - R::pbeta(xLower, alpha, gamma,1, 0) )<0 ) DBG4("error_minus"<<endl)
    //if(probSd!=probSd)  DBG4(endl << " Error" << " lower=" << lower << " upper=" << upper << " centerp=" << var <<" t: "<<t<< endl)
    return (probSd);  // Rf_pgamma(q, shape, scale, lower.tail, log.p)
}


// ===================================================

int HMDP::countStatesHMDP(){
  int op,y;
  y=0;
  for(op=0;op<opNum;op++){
    if(op==0) y=1; else y=y+IdCountL1(op-1);
    y=y+IdCountL2(op);
  }
  return y + IdCountL1(opNum-1) +1;
}


// ===================================================

int HMDP::IdCountL2(int op){
  int x, t, iMW, iSW, iMP, iSP, iT, iP, opS, d;
  x=0;

  for(t=opE[op]; t<=opL[op]; t++) {
    for(opS=opE[op];opS<=t;opS++){
      if( (op==0) & (opS!=1) ) continue;
      if(opS>opL[op]-opD[op]) continue;
      for(d=0; d<=opD[op]; d++){
        if( opD[op]-t+opS>d ) continue;
        for(iMW=0; iMW<sizeSMW; iMW++){
          for(iSW=0; iSW<sizeSSW; iSW++){
            for(iMP=0; iMP<sizeSMP; iMP++){
              for(iSP=0; iSP<sizeSSP; iSP++){
                for(iT=0; iT<sizeST; iT++){
                  for(iP=0; iP<sizeSP; iP++){
                    x++;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return(x);
}

// ===================================================

int HMDP::IdCountL1(int op){
  int x, iMW, iSW, iMP, iSP, iT, iP, opS;
  x=0;

  for(opS=opE[op]; opS<=opL[op]; opS++){
    if( (op==0) & (opS!=1) ) continue;
    if(opS>opL[op]-opD[op]) continue;
    for(iMW=0; iMW<sizeSMW; iMW++){
      for(iSW=0; iSW<sizeSSW; iSW++){
        for(iMP=0; iMP<sizeSMP; iMP++){
          for(iSP=0; iSP<sizeSSP; iSP++){
            for(iT=0; iT<sizeST; iT++){
              for(iP=0; iP<sizeSP; iP++){
                x++;
              }
            }
          }
        }
      }
    }
  }
  return(x+2);
}

// ===================================================

