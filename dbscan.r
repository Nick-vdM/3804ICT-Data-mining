library(fpc)

disSim <- read.csv('E:/Users/nicol/Documents/GitHub/3804ICT-Data-mining-try-4/pickles/dis_sim.csv')
# R is just failing to print this so 
# I think our data is just too big

dbscan_cl <- dcscan(disSim, eps=0.75, minPts=20 method=c("dist"))
# Now it's properly just stuck