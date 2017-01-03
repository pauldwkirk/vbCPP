vbgmm <- function(X,K,maxIter = 2000, tol = 1e-8)
{
  results  <- vbCPP(X,K,maxIter, tol) 
  return(results)
}