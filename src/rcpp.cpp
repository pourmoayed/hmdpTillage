#include "hmdp.h"

using namespace Rcpp;
using namespace std;

//' Build the HMDP (3 levels with shared linking) using the C++ binary writer.
//'
//' The MDP specified at level 3 is generated for each tillage operation for a dummy state at the first stage at level 2.
//'
//' @param filePrefix Prefix used by the binary files storing the MDP model.
//' @param paramModel parameters a list created using \code{\link{setParameters}}.
//'
//' @return Build log (character).
//' @export
// [[Rcpp::export]]
SEXP BuildHMDP(const CharacterVector filePrefix, const List paramModel) {
   string prefix = as<string>(filePrefix);
   HMDP Model(prefix, paramModel);
   Rcout << "Total number of states: " << Model.countStatesHMDP() << endl;
   return( Model.BuildHMDP() );
   //return(wrap(0));
}













