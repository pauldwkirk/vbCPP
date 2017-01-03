\name{vbgmm}
\alias{vbgmm}

\title{Perform variational Bayes mixture of Gaussians}
\usage{
vbgmm(X, K, maxIter=2000, tol=1e-8)
}
\arguments{
  \item{X}{data matrix. Columns are observations, rows covariates.}
  
  \item{K}{number of clusters.}

  \item{maxIter}{maximum number of iterations.  The default is
                2000.}

\item{tol}{tolerance. The default is 1e-8.}

}
\description{
  \code{vbgmm} performs variational Bayes GMM.
}

\value{
  \code{vbgmm} returns a list of outputs.
}

\details{Performs variational Bayes GMM.
}

\author{
  Paul Kirk (\url{http://www.mrc-bsu.cam.ac.uk/people/paul-kirk/}).
}
\seealso{}

\examples{
# load vbCPP library
library("vbCPP")

# run code
set.seed(1)
K      <- 3
data("X")
result <- vbgmm(X,K)



}
\keyword{Mixture modelling}
