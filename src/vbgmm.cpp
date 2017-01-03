# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;


arma::rowvec rowvecdigamma(arma::rowvec v){
  arma::rowvec outvec(v.n_elem);
  for(int k = 0; k<v.n_elem; k++){
    outvec(k) = R::digamma(v(k));
  }
  return outvec;
}

arma::rowvec rowveclgammaf(arma::rowvec alpha){
  arma::rowvec outvec(alpha.n_elem);
  for(int k = 0; k<alpha.n_elem; k++){
    outvec(k) = lgamma(alpha(k));
  }
  return outvec;
}

arma::mat matlgammaf(arma::mat A){
  arma::mat outMat(A.n_rows, A.n_cols);
  for(int k = 0; k<A.n_rows; k++){
    outMat.row(k) = rowveclgammaf(A.row(k));
  }
  return outMat;
}

double logMvGamma(double x, int d)
{
  arma::rowvec xvec(d);
  double y;
  for(int k = 0; k<d; k++){
    xvec(k) = x - (k/2.0);
  }
  y = (d*(d-1)/4.0)*log(arma::datum::pi)+sum(rowveclgammaf(xvec));
  return y;
}

arma::rowvec logMvGammaVec(arma::rowvec x, int d)
{
  arma::mat xmat(d, x.n_elem);
  arma::rowvec y(x.n_elem);
  for(int k = 0; k<d; k++){
    for(int j = 0; j< x.n_elem; j++)
    {
      xmat(k, j) = x(j) - (k/2.0);
    }
  }
  y = (d*(d-1)/4.0)*log(arma::datum::pi)+sum(matlgammaf(xmat));
  
  return y;
}

arma::mat matrixdigamma(arma::mat A){
  arma::mat outMat(A.n_rows, A.n_cols);
  for(int k = 0; k<A.n_rows; k++){
    outMat.row(k) = rowvecdigamma(A.row(k));
  }
  return outMat;
}

arma::vec logsumexp(arma::mat X){
  arma::vec y(X.n_rows);
  arma::vec s(X.n_rows);
  arma::mat Xnew(size(X));
  
  // subtract the largest in each row
  y  = arma::max(X,1);
  Xnew = X;
  Xnew.each_col() -= y;
  Xnew = exp(Xnew);
  s    = y + log(sum(Xnew, 1));
  return s;
}

class multivariateGaussianPrior {
public:
  double alpha; //
  double beta;  //
  double v;     //
  double logW;  //
  arma::vec m;  //
  arma::mat M;  //
  
  // Constructor to set dimensions & defaults:
  multivariateGaussianPrior(const arma::mat X, const int d)
    : m(d)
    , M(d, d)// matrix member initialization
  {
    M.eye();
    arma::mat cholM   = chol(M);
    
    alpha = 1;
    beta  = 1;
    v     = d + 1;
    m     = mean(X,1); 
    logW  = -2*sum(log(cholM.diag()));
  }

    
  
};


class multivariateGaussianModel {
public:
  int K;
  int n;
  int d;
  double L;
  
  arma::cube U;
  arma::rowvec alpha; //
  arma::rowvec beta; //
  arma::rowvec v;     //
  arma::rowvec logW;
  arma::mat m;     //
  arma::mat M;  //
  arma::mat R;  //
  arma::mat logR;  //
  
  // Constructor to set dimensions & initialise:
  multivariateGaussianModel(int nn, int KK, int dd)
    : U(dd, dd, KK)
    , alpha(KK)
    , beta(KK)
    , v(KK)
    , logW(KK)
    , m(dd, KK)
    , M(dd, dd)
    , R(nn, KK)
    , logR(nn, KK)
    // matrix member initialization
  {
    n = nn;
    K = KK;
    d = dd;
    arma::vec labels = arma::randi<arma::vec>( n , arma::distr_param(0,K-1) );
    //labels = arma::randi<arma::vec>( n , arma::distr_param(0,K-1) );
    R.zeros();  
    
    for(int k = 0; k < K; k++ ){
      R.elem( find(labels == k) + (k*n) ).ones();  
    }
  }
  
  
  void maximize(const arma::mat X, multivariateGaussianPrior prior)
  {
    arma::rowvec nk  = sum(R); // 10.51
    alpha            = prior.alpha+nk; // 10.58
    beta             = prior.beta+nk; // 10.60
    v                = prior.v+nk; // 10.63
    
    m                = X*R;
    m.each_col()    += prior.beta*prior.m;
    m.each_row()    /= beta;// 10.61
    
    logW.zeros();
    U.zeros();
    arma::mat r = sqrt(R.t());
    arma::mat Xm(d,n);
    arma::vec m0m(K);
    arma::mat cholM(d,d);
    
    for(int i =0; i < K; i++)
    {
      Xm  = X;
      Xm.each_col()  -= m.col(i);
      Xm.each_row()  %= r.row(i);
      m0m = prior.m - m.col(i);
      M   = prior.M + (Xm*Xm.t()) + (prior.beta*(m0m*m0m.t()));
      cholM = chol(M);
      U.slice(i) = cholM;
      logW(i) = -2*sum(log(cholM.diag()));
    }

  }
  
  
  void expect(const arma::mat X)
  {
    
    arma::mat EQ(n, K);
    arma::mat logRho(n, K);
    arma::mat Q(d,n);

    arma::mat cholM(d,d);
    arma::mat Xm(d,n);

    arma::rowvec EQcol(n);
    arma::rowvec ElogLambda(K); 
    arma::rowvec Elogpi(K); 
    
    
    EQ.zeros();
    
    
    for(int i = 0; i < K; i++)
    {
      cholM = U.slice(i);
      Xm    = X;
      Xm.each_col() -= m.col(i);
      Q = solve(cholM.t(), Xm, arma::solve_opts::fast);
      EQcol = (d/beta(i)) + v(i)*(sum(Q%Q));
      EQ.col(i) = EQcol.t();
    }
    
    arma::mat repV(d, K);
    for(int j = 0; j < d; j++)
    {
      repV.row(j) = v - j;
    }
    repV               = matrixdigamma( repV/2 );
    ElogLambda         = sum(repV) + (d*log(2)) + logW;
    Elogpi             = rowvecdigamma(alpha) - R::digamma(sum(alpha));
    logRho             = EQ;
    logRho.each_row() -= (ElogLambda-(d*log(2*(arma::datum::pi))));
    logRho             = -logRho/2;
    logRho.each_row() += Elogpi;
    logR               = logRho;
    

    logR.each_col()   -= logsumexp(logRho);
    R                  = exp(logR);
  }
  
  
  void bound(const arma::mat X, multivariateGaussianPrior prior)
  {
    double Epz        = 0;
    double Eqz        = accu(R%logR);
    double logCalpha0 = lgammaf(K*prior.alpha) - K*lgammaf(prior.alpha); 
    double logCalpha  = lgammaf(sum(alpha)) - sum(rowveclgammaf(alpha));
    double Epmu       = 0.5*d*K*log(prior.beta);
    double Eqmu       = 0.5*d*sum(log(beta));
    double logB0      = -0.5*prior.v*(prior.logW+d*log(2))-logMvGamma(0.5*prior.v,d);
    double EpLambda   = K*logB0;
    arma::rowvec logB = -0.5*v%(logW+d*log(2))-logMvGammaVec(0.5*v,d);
    double EqLambda   = sum(logB);
    double EpX        = -0.5*d*n*log(2*(arma::datum::pi));
    L                 = Epz-Eqz+logCalpha0-logCalpha+Epmu-Eqmu+EpLambda-EqLambda+EpX;
    
}
  
  
};

// [[Rcpp::export()]]
List vbCPP (const arma::mat X, const int K, const int maxIter = 2000, double tol = 1e-8) {
  
  const int n = X.n_cols;
  const int d = X.n_rows;
  int i;
  arma::vec L(maxIter);
  arma::vec outputL;
  
  multivariateGaussianPrior prior(X, d);
  multivariateGaussianModel model(n, K, d);
  
  model.maximize(X, prior);

  L.fill(-arma::datum::inf);
  for(i = 1; i < maxIter; i++)
  {
    model.expect(X);
    model.maximize(X, prior);
    model.bound(X, prior);
    L(i) = model.L/n;
    if(fabs(L(i) - L(i-1))  < tol*fabs(L(i))){
      break;
    }
  }

  arma::ucolvec labels  = index_max( model.R, 1) + 1;
  outputL               = L.subvec(1, i);  

  // returns
  arma::rowvec dig = rowvecdigamma(model.alpha);
  List ret ;
  ret["R"]       = model.R;
  ret["alpha"]   = model.alpha;
  ret["beta"]    = model.beta;
  ret["m"]       = model.m;
  ret["v"]       = model.v;
  ret["U"]       = model.U;
  ret["logW"]    = model.logW;
  ret["logR"]    = model.logR;
  ret["L"]       = outputL;
  ret["labels"]  = labels;
  
  
  return(ret) ;
}
