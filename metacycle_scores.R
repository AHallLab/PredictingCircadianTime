library(MetaCycle)
setwd("C:\\Users\\colmer\\Downloads\\PredictingCircadianTime-main (2)\\PredictingCircadianTime-main")

meta2d(infile="Data//X_train_times.csv", filestyle="csv", outdir="MetaScores", 
       timepoints="Line1", cycMethod=c("JTK","ARS"), 
       outIntegration="noIntegration", nCores=8)
