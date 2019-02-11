## Matrix Decompositions in Data Analysis
## Winter 2019
## Assignment file
## FILL IN the following information
## Name:
## Student ID:

##
## This file contains stubs of code to help you to do your 
## assignment. You can fill your parts at the indicated positions
## and return this file as a part of your solution. 
##
## Remember to fill your name and student ID number above.
##


## Task 1
##########

## Boilerplate for ALS NMF
nmf.als <- function(A, k, maxiter=300) {
	result <- NA
	# Initialize W and H (only one is actually needed)
	W <- matrix(runif(nrow(A)*k), nrow(A), k)
	H <- matrix(runif(k*ncol(A)), k, ncol(A))
	errs <- rep(NA, times=maxiter) # Stores errors per iteration
	for (i in 1:maxiter) {
		# FILL IN update for W and H
		errs[i] <- norm(A - W%*%H, 'F')
	}
	# Copy the column names to H; helps analysing it
	colnames(H) <- colnames(A)
	result <- list(W=W, H=H, errs=errs)
	result
}

## Load the news data
A <- as.matrix(read.csv("news.csv"))

## Sample use of nmf.als:
res <- nmf.als(A, 20) # Computes NMF with ALS updates
## To show the per-iteration error
plot(res$errs, type="l", main="Convergence of NMF ALS", xlab="Iteration", ylab="Squared Frobenius")


## IMPLEMENT the other algorithms
## DO the comparisons


#############
## Task 2
#############

## Remember to normalize the data before applying the NMF algorithms
B = A/sum(A)
## If res <- nmf.als(A, 20, ...), you can print the top-10 terms by
for (k in 1:nrow(res$H)) print(sort(res$H[k,], decreasing=TRUE)[1:10])

## USE NMF to analyse the data
## REPEAT the analysis with GKL-optimizing NMF

#############
## Task 3
#############

## General code to compute the NMI
nmi <- function(x, opt) {
  p.x <- table(x)/length(x)
  p.y <- table(opt)/length(opt)
  if (length(p.x) != length(p.y)) {
    stop("Data has incorrect number of clusters")
  }
  p.xy <- table(x, opt)/length(x)

  H.xgy <- 0
  H.ygx <- 0
  H.xy <- 0
  for (i in 1:length(p.y)) {
    for (j in 1:length(p.y)) {
      if (p.xy[i,j] > 0) {
        H.xgy <- H.xgy + p.xy[i,j]*log(p.x[i]/p.xy[i,j])
        H.xy <- H.xy - p.xy[i,j]*log(p.xy[i,j])
      }
      if (p.xy[j,i] > 0) {
        H.ygx <- H.ygx + p.xy[j,i]*log(p.y[j]/p.xy[j,i])
      }
    }
  }
  I <- H.xy - H.xgy - H.ygx
  as.vector((H.xy - I)/H.xy)
}
  
## Code to compute the NMI for the news
nmi.news <- function(x) { nmi(x, as.matrix(read.csv("news_ground_truth.txt",header = FALSE))) }

## Here we compute Karhunen--Loeve
Z <- scale(A) # Z-scores
SVD <- svd(A, nu=0, nv=20)
V = SVD$v
KL = Z%*%V

## COMPUTE pLSA with the matrix B from the previous task

## Do some clustering, here for the KL matrix
clust <- kmeans(KL, 20, iter.max=100, nstart=20)$cluster

## How good is this? The smaller the better
nmi.news(clust)

## DO NMF with GKL optimization and other comparisons
