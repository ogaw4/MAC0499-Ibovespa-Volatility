library(dplyr)
library(ggplot2)
library(reshape2)
library(glue)

dset_path <- "data/input_vol5d.csv"
res_path <- "data/results5dv2.csv"

train_max_idx <- 2064
valid_max_idx <- 2305


dset <- read.csv(dset_path) %>% mutate(DATE = as.Date(as.character(DATE)))
res <- read.csv(res_path)

lab <- c(rep("train", train_max_idx), rep("validation", valid_max_idx - train_max_idx), rep("real", nrow(dset) - valid_max_idx))

dset$lab <- lab
dset <- na.omit(dset)

res$lab <- "predicted"
res$DATE <- dset$DATE


.df <- data.frame(ref_date = c(dset$DATE, res$DATE),
                  value = c(dset$VOL20D5P, res$pred),
                  label = c(dset$lab, res$lab))


ggplot(.df, aes(x = ref_date, y = value, colour = label)) + geom_line(size = 1) +
  ggtitle("D+5 IBOVESPA Volatility")


res$pred[1:valid_max_idx] <- NA

.df <- data.frame(ref_date = c(dset$DATE, res$DATE),
                  value = c(dset$VOL20D5P, res$pred),
                  label = c(dset$lab, res$lab))
ggplot(.df %>% filter(ref_date >= "2015-01-01"), aes(x = ref_date, y = value, colour = label)) + geom_line(size = 1) +
  ggtitle("D+5 IBOVESPA Volatility (Zoomed in)")
