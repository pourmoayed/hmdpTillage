
#ifndef HMDP_HPP
#define HMDP_HPP

#include "RcppArmadillo.h"    // we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "binaryMDPWriter.h"
#include "time.h"

using namespace Rcpp;
using namespace std;

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do

// [[Rcpp::depends(RcppArmadillo)]]

// ===================================================

/**
* Class for building a 3 level HMDP modelling tillage operations.
*
* It store all the info needed for writing the HMDP to the binary files.
*
* @author Reza Pourmoayed
*/
class HMDP
{
public:  // methods

    /** Constructor. Store the parameters.
     *
     * @param filePrefix Prefix used by the binary files storing the MDP model.
     * @param paramModel A list for model parameters related to HMDP and SSMs created using \code{setParam} in R.
     */
   HMDP(const string filePrefix, const List paramModel);


   /** Build the HMDP (to binary files). Use "shared linking".
    *
    *  Build a 3 level HMDP saved in binary files by using the
    *  binaryMDPWriter (c++ version).
    *
    *  @return Build log (string)
    *  @export
    */
   SEXP BuildHMDP();


   /** Count the number of states in the HMDP */
   int countStatesHMDP();


private:

   /** Compare two vectors and return an error if not equal. */
   template <typename T>
   bool inline compareVec(const vector<T>& v1, const vector<T>& v2) {
      if (v1.size()!=v2.size()) {
         Rcout << "Vectors do not have the same size!" << endl;
         Rcout << "v1: " << vec2String<T>(v1) << " v2: " << vec2String<T>(v2) << endl;
         return false;
      }
      for (int i=0;i<v1.size(); i++) {
         if(!Equal(v1[i],v2[i])) {
            return false;
         }
      }
      return true;
   }

   /** Compare all input vectors with tmp vectors*/
   void compareAllVec(string state, string action, const vector<int> & idxTmp, const vector<int> & scpTmp, const vector<flt> prTmp) {
      bool error = false;
      if (!compareVec<flt>(prTmp,pr)) error = true;
      if (!compareVec<int>(idxTmp,index)) error = true;
      if (!compareVec<int>(scpTmp,scope)) error = true;
      if (error) {
         Rcout << "Error when comparing vectors in state " << state << " and action " << action << ":\n";
         Rcout << "idxTmp: " << vec2String<int>(idxTmp) << endl << " index: " << vec2String<int>(index) << endl;
         Rcout << "scpTmp: " << vec2String<int>(scpTmp) << endl << " scope: " << vec2String<int>(scope) << endl;
         Rcout << "prTmp: " << vec2String<flt>(prTmp) << endl << " pr: " << vec2String<flt>(pr) << endl;
         Rcpp::stop("");
      }
   }


   /** Calculate and fill arrays with rewards and trans pr. */
   void Preprocess();


   /** Create the process at level 1. */
   void BuildL1Process();


   /** Create the stages at level 1. */
   void BuildL1Operation();


   /** Create last stage of level 1 (dummy stage). */
   void BuildL1StageLast();


   /** Create the process at level 2 for a specific tillage operation.
    *
    * @param op Tillage operation under consideration.
    */
   void BuildL2Process(int op);


   /** Calculate the initial transition probabilities and weights for the first day.
    *
    * Set the class vectors scope, pr and index based on global vector iniDist.
    */
   void WeightTransPrIni();


   /** Calculate the reward and transition probabilities for action post. related to postpone tillage operation.
    *
    * @param op Tillage operation under consideration
    * @param opSt Index of state for finishing time of operation op-1 (op>1)
    * @param dt Index of state for remaining days for finishing operation op
    * @param iMWt Index of state for estimated mean of soil water content
    * @param iSWt Index of state for estimated standard deviation of soil water content
    * @param iMPt Index of state for estimated posterior mean of latent variable in Gaussian SSM (error factor)
    * @param iSPt Index of state for estimated posterior sandard deviation of latent variable in Gaussian SSM (error factor)
    * @param iTt Index of state for weather forecast regarding air temprature.
    * @param iPt Index of state for weather forecast regarding precipitation.
    * @param t Current day.
    *
    */
   void WeightsTransPos(int & op, int & opSt, int & dt, int & iMWt, int & iSWt, int & iMPt, int & iSPt, int & iTt, int & iPt, int & t);


   /** Calculate the reward and transition probabilities for action do. related to performing a tillage operation.
    *
    * @param op Tillage operation under consideration
    * @param opSt Index of state for finishing time of operation op-1 (op>1)
    * @param dt Index of state for remaining days for finishing operation op
    * @param iMWt Index of state for estimated mean of soil water content
    * @param iSWt Index of state for estimated standard deviation of soil water content
    * @param iMPt Index of state for estimated posterior mean of latent variable in Gaussian SSM (error factor)
    * @param iSPt Index of state for estimated posterior sandard deviation of latent variable in Gaussian SSM (error factor)
    * @param iTt Index of state for weather forecast regarding air temprature.
    * @param iPt Index of state for weather forecast regarding precipitation.
    * @param t Current day.
    *
    */
   void WeightsTransDo(int & op, int & opSt, int & dt, int & iMWt, int & iSWt, int & iMPt, int & iSPt, int & iTt, int & iPt, int & t);


   /** Calculate the reward and transition probabilities for action term. related to finishing  a tillage operation.
    *
    * @param op Tillage operation under consideration
    * @param opSt Index of state for finishing time of operation op-1 (op>1)
    * @param dt Index of state for remaining days for finishing operation op
    * @param iMWt Index of state for estimated mean of soil water content
    * @param iSWt Index of state for estimated standard deviation of soil water content
    * @param iMPt Index of state for estimated posterior mean of latent variable in Gaussian SSM (error factor)
    * @param iSPt Index of state for estimated posterior sandard deviation of latent variable in Gaussian SSM (error factor)
    * @param iTt Index of state for weather forecast regarding air temprature.
    * @param iPt Index of state for weather forecast regarding precipitation.
    * @param t Current day.
    *
    */
   void WeightsTransTerm(int & op, int & opSt, int & dt, int & iMWt, int & iSWt, int & iMPt, int & iSPt, int & iTt, int & iPt, int & t);


   /** Calculate the reward and transition probabilities for action skip. related to skip tillage operations for the current cropping cycle.
    *
    *@param op Tillage operation under consideration
    *
    */
   void WeightsTransSkip(int & op);


   /** Build a map of the index of states in the second level of HMDP. That is, the map to identify state id in the second level.
    *
    * @param op Tillage operation under consideration.
    */
   void BuildMapL1Vector(int op);


   /** Build a map of the index of states for the given stage/time in the third level. That is, the map to identify state id in the third level given the current stage and tillage operation.
    *
    * @param t current day.
    * @param op Tillage operation under consideration.
    *
    */
   void BuildMapL2Vector(int t, int op);


   /** Calculate the reward values under action Do.
    *
    *  Values are stored in the vector \var(rewDo[op][iMW][iSW]).
    */
   void CalcRewaerdDo();

   /** Calculate the transition probability values for estimated mean of soil water content.
    *
    *  Values are stored in the vector \var(prMW[iMWt][iMPt][iSPt][iTt][iPt][iMW]).
    */
   void CalcTransPrMW();


   /** Calculate the transition probability values for estimated standard deviation of soil water content.
    *
    *  Values are stored in the vector \var(prSW[t][iSWt][iSW]).
    */
   void CalcTransPrSW();


   /** Calculate the transition probability values for posterior mean of latent variable (error factor) in Gaussian SSM.
    *
    *  Values are stored in the vector \var(prMP[iMPt][iSPt][iMP]).
    */
   void CalcTransPrMP();


   /** Calculate the transition probability values for posterior standard deviation of latent variable (error factor) in Gaussian SSM.
    *
    *  Values are stored in the vector \var(prSP[iMWt][iSPt][iTt][iPt][iSP]).
    */
   void CalcTransPrSP();


   /** Calculate the transition probability values for weather information regarding air temprature.
    *
    *  Values are stored in the vector \var(prT[iTt][iPt][iT]).
    */
   void CalcTransPrT();


   /** Calculate the transition probability values for weather information regarding precipitation.
    *
    *  Values are stored in the vector \var(prP[iPt][iP]).
    */
   void CalcTransPrP();


   /** Calculate the future soil water content based on a rainfall-runoff model given in \url(http://onlinelibrary.wiley.com/doi/10.1002/hyp.6629/abstract).
    *
    * @param Wt The current Soil water content (Volumetric measure)
    * @param Tt Prediction of average air temprature for the next day.
    * @param Pt Prediction of total precipitation for the next day.
    *
    * @return A predition of soil water content at the next day.
    */
   double Hydro(double & Wt, double & Tt, double Pt);


   /** Calculate the transition probability for variance component (posterior mean of latent variable) in a non-Gaussian SSM baesd on Theorem 4 in \url(http://www.sciencedirect.com/science/article/pii/S0377221715008802).
    *
    * @param t current day.
    * @param lower Lower limit of a variance component in the next day.
    * @param upper Upper limit of a variance component in the next day.
    * @param var Center point of the variance component in the current day.
    *
    * @return Transition probability for the variance component at the next day given variance var.
    */
   double PrNGSSM(int t, double lower, double upper, double var);


    /** Count the number of states in level 1.
    */
    int IdCountL1(int op);


    /** Count the number of states in level 2.
    */
    int IdCountL2(int op);


   /** Convert integers into a string. */
   string getLabel(const int & a, const int & b) {
      std::ostringstream s;
      s << "(" << a << "," << b << ")";
      return s.str();
   }
   /** Convert integers into a string. */
   string getLabel(const int & a, const int & b, const int & c) {
      std::ostringstream s;
      s << "(" << a << "," << b << "," << c << ")";
      return s.str();
   }
   /** Convert integers into a string. */
   string getLabel(const int & a, const int & b, const int & c, const int & d) {
      std::ostringstream s;
      s << "(" << a << "," << b << "," << c << "," << d << ")";
      return s.str();
   }
   /** Convert integers into a string. */
   string getLabel(const int & a, const int & b, const int & c, const int & d, const int & e) {
      std::ostringstream s;
      s << "(" << a << "," << b << "," << c << "," << d << "," << e << ")";
      return s.str();
   }

   /** Convert integers into a string. */
   string getLabel(const int & a, const int & b, const int & c, const int & d, const int & e, const int & f) {
      std::ostringstream s;
      s << "(" << a << "," << b << "," << c << "," << d << "," << e << "," << f << ")";
      return s.str();
   }

   /** Convert integers into a string. */
   string getLabel(const int & a, const int & b, const int & c, const int & d, const int & e, const int & f, const int & g) {
      std::ostringstream s;
      s << "(" << a << "," << b << "," << c << "," << d << "," << e << "," << f << "," << g << ")";
      return s.str();
   }

         /** Convert integers into a string. */
   string getLabel(const int & a, const int & b, const int & c, const int & d, const int & e, const int & f, const int & g, const int & h) {
      std::ostringstream s;
      s << "(" << a << "," << b << "," << c << "," << d << "," << e << "," << f << "," << g << "," << h << ")";
      return s.str();
   }

   /** Convert integers into a string. */
   string getLabel(const int & a, const int & b, const int & c, const int & d, const int & e, const int & f, const int & g, const int & h, const int & k) {
     std::ostringstream s;
     s << "(" << a << "," << b << "," << c << "," << d << "," << e << "," << f << "," << g << "," << h << "," << k << ")";
     return s.str();
   }

   /** Convert integers into a string. */
   string getLabel(const int & a, const int & b, const int & c, const int & d, const int & e, const int & f, const int & g, const int & h, const int & k, const int & z) {
     std::ostringstream s;
     s << "(" << a << "," << b << "," << c << "," << d << "," << e << "," << f << "," << g << "," << h << "," << k << "," << z << ")";
     return s.str();
   }

//----------------------------------------------------------------------------------------------------------------------------------

private:   // variables

  static const double ZERO;  // trans pr below are considered as ZERO

  arma::vec opSeq;
  arma::vec opE;
  arma::vec opL;
  arma::vec opD;
  arma::vec opDelay;
  arma::vec opFixCost;
  arma::vec watTh;

  int opNum;
  int tMax;
  int idS;
  double coefLoss;
  double priceYield;
  double machCap;
  double yieldHa;
  double fieldArea;
  double coefTimeliness;
  double costSkip;

  double temMeanDry;
  double temMeanWet;
  double temVarDry;
  double temVarWet;
  double dryDayTh;
  double precShape;
  double precScale;
  double prDryWet;
  double prWetWet;

  double hydroWatR;
  double hydroWatS;
  double hydroM;
  double hydroKs;
  double hydroLamba;
  double hydroETa;
  double hydroETb;
  double hydroETx;

  double gSSMW;
  double gSSMV;
  double gSSMm0;
  double gSSMc0;
  double nGSSMm0;
  double nGSSMc0;
  double nGSSMK;

  bool check;

  arma::mat dMP;
  arma::mat dSP;
  arma::mat dMW;
  arma::mat dSW;
  arma::mat dT;
  arma::mat dP;

  arma::vec sMP;
  arma::vec sSP;
  arma::vec sMW;
  arma::vec sSW;
  arma::vec sT;
  arma::vec sP;

  int sizeSMP;
  int sizeSSP;
  int sizeSMW;
  int sizeSSW;
  int sizeST;
  int sizeSP;


  // variables used when build process
   vector<int> scope;
   vector<int> index;
   vector<flt> pr;
   vector<flt> weights;

   map<string,int> mapL1;   // map to identify state id at level 1 given string (t,n,iSW,iSSd)
   map<string,int> mapR;    // find unique id for a state at level 2 (whole process)
   //vector<vector<vector< vector< vector<int> > > > > mapL2Vector; // mapL2Vector[RS][n][iSW][iSG][iSSd] Vector of mapL2 instead of strings to speed up the codes in the second level


   vector <vector<vector< vector< vector< vector<double> > > > > > prMW;
   vector< vector< vector<double> > > prMP;
   vector<vector< vector< vector< vector<double> > > > > prSP;
   vector< vector< vector<double> > > prSW;
   vector <vector< vector<double> > > prT;
   vector< vector<double> > prP;
   vector <vector< vector<double> > > rewDo;
   vector< vector< vector< vector< vector< vector< vector< vector<int> > > > > > > > mapL2Vector;
   vector< vector< vector< vector< vector< vector< vector<int> > > > > > > mapLVector;

   string label;
   binaryMDPWriter w;
   ostringstream s;         // stream to write labels

   TimeMan cpuTime;
};


#endif
