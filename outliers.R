library(tidyverse)
library(Hmisc)

df <- read_csv("data/train_df.csv")


ggplot(df, aes(y = LotArea))+ geom_boxplot() + theme_bw()

ggsave("boxplot_lotarea.pdf", device = "pdf", dpi =  "retina", 
       height = 6, width = 3, units = "in")

ggplot(df, aes(y = TotalBsmtSF))+ geom_boxplot() + theme_bw()

ggsave("boxplot_totalmbsmt.pdf", device = "pdf", dpi =  "retina", 
       height = 6, width = 3, units = "in")


boxplot(df$TotalBsmtSF)

describe(df$LotArea)
describe(df$TotalBsmtSF)

df %>% filter(LotArea > 80000)%>% select(y_train)
df %>% filter(TotalBsmtSF > 6000) %>% select(y_train)

