


###############
# A) load the data only weekly returns
###############

# Packes required for subsequent analysis. P_load ensures these will be installed and loaded. 
if (!require("pacman")) install.packages("pacman")
pacman::p_load(quantmod,
               keras,
               tensorflow)

# run this only if using for first time
install_tensorflow()

# parameters - get data for the past 10 years, 2020-2010, june 1st. 
end <- "2020-06-01"
start <- "2010-06-01"

# these are stock tickers
symetf <- c("IEF", "TLT", "SPY", "QQQ")

# pulls the data from the yahoo finance API, turns returns weekly
l <- length(symetf)
w0 <- NULL
for (i in 1:l) {
  dat0 <- getSymbols(symetf[i],
                     src = "yahoo", from = start, to = end, auto.assign = F,
                     warnings = FALSE, symbol.lookup = F
  )
  w1 <- weeklyReturn(dat0)
  w0 <- cbind(w0, w1)
}

# turn data on weekly returns to matrix, set row and column names
dat <- as.matrix(w0)
timee <- as.Date(rownames(dat))
dat <- na.fill(dat, fill = "extend")
colnames(dat) <- symetf

# write to csv
write.csv(dat,"stock_data_q1.csv")


###############
# B) apply PCA
###############


# Get results PCA - we do not standardize before applying it
PCA_result_weeklyReturns <- prcomp(dat)


# get variance explained per principal component
var_explained <- PCA_result_weeklyReturns$sdev^2/sum(PCA_result_weeklyReturns$sdev^2)


##############
# C) Create an auto-encoder 
#############

# get the input size - number of features (e.g. stocks)
input_size = ncol(dat)

# how many features do we want to represent the data?
encoding_size = 2 

## The auto-encoder exists of two parts

# first, the encoder
enc_input = layer_input(shape = input_size)

enc_output = enc_input %>%
  layer_dense(units =encoding_size, activation='linear') # use linear activation function
encoder = keras_model(enc_input, enc_output)

# second, the decoder
dec_input = layer_input(shape = encoding_size)
dec_output = dec_input %>%
  layer_dense(units=input_size, activation = "linear")
decoder = keras_model(dec_input, dec_output)

# combine encoder and decoder into auto-encoder
autoEncoder_input = layer_input(shape = input_size)
autoEncoder_output = autoEncoder_input %>%
  encoder() %>%
  decoder()
autoEncoder = keras_model(autoEncoder_input, autoEncoder_output)
summary(autoEncoder)


# learn the representation
autoEncoder %>% compile(optimizer="adam", loss="mse")
autoEncoder %>% fit(dat, dat, epochs = 10, batch_size = 1)


##############
# D) get result of auto encoder, compare to PCA
#############

# get principal component for N observations 
principal_component_1 = PCA_result_weeklyReturns$x[,1]
principal_component_2 = PCA_result_weeklyReturns$x[,2]


encoded_data = encoder %>% predict(dat)
decoded_data = decoder %>% predict(encoded_data)

decoded_data - dat
