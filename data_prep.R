## 
library(tidyverse)
library(magrittr)
library(reticulate)

dat <- read.table("Data/apsr.tab", header = T, sep = "\t", fill = TRUE)
dat %<>% rename(y = disp, similarity = sq, asymetry = aysm, peace_years = py, contiguous = contig)



dat86 <- filter(dat, year < 1986)
dat89 <- filter(dat, year >= 1986)

dat %<>% select(-year)
dat86 %<>% select(-year)
dat89 %<>% select(-year)

dat %<>% r_to_py()
dat86 %<>% r_to_py()
dat89 %<>% r_to_py()

source_python("lstm_validation.py")







