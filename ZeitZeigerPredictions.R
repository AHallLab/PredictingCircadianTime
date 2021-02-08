library('doParallel')
library('dplyr')
library('ggplot2')
library('tidyr')
library('zeitzeiger')

x_train <- read.csv(file = 'Data\\X_train_raw.csv')
x_train = data.matrix(x_train)
y_train <- read.csv(file = 'Data\\Y_train.csv')
y_train = data.matrix(y_train)
y_train = y_train / 24

x_valid <- read.csv(file = 'Data\\X_valid_raw.csv')
x_valid = data.matrix(x_valid)
y_valid <- read.csv(file = 'Data\\Y_valid.csv')
y_valid = data.matrix(y_valid)
y_valid = y_valid / 24

x_test <- read.csv(file = 'Data\\X_test_raw.csv')
x_test = data.matrix(x_test)
y_test <- read.csv(file = 'Data\\Y_test.csv')
y_test = data.matrix(y_test)
y_test = y_test / 24

zzFit = zeitzeiger(t(x_train), y_train, t(x_train), sumabsv = 1, nSpc = 2)
results = zzFit['predResult']
predResult = results[["predResult"]]
error = getCircDiff(predResult$timePred, t(y_train))
error = unlist(error, use.names = FALSE)
error = abs(error)
error1 = mean(error)
dfTest = data.frame(timeObs = t(y_train), timePred = predResult$timePred, timeError = getCircDiff(predResult$timePred, t(y_train)))
ggplot(dfTest) +
  geom_point(aes(x = timeObs, y = timePred), size = 2, shape = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed') +
  scale_x_continuous(limits = c(0, 1)) + scale_y_continuous(limits = c(0, 1)) +
  labs(x = 'Observed time', y = 'Predicted time') + theme_bw()
print(error1)

zzFit = zeitzeiger(t(x_train), y_train, t(x_valid), sumabsv = 1, nSpc = 2)
results = zzFit['predResult']
predResult = results[["predResult"]]
error = getCircDiff(predResult$timePred, t(y_valid))
error = unlist(error, use.names = FALSE)
error = abs(error)
error1 = mean(error)
dfTest = data.frame(timeObs = t(y_valid), timePred = predResult$timePred, timeError = getCircDiff(predResult$timePred, t(y_valid)))
ggplot(dfTest) +
  geom_point(aes(x = timeObs, y = timePred), size = 2, shape = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed') +
  scale_x_continuous(limits = c(0, 1)) + scale_y_continuous(limits = c(0, 1)) +
  labs(x = 'Observed time', y = 'Predicted time') + theme_bw()
print(error1)

zzFit = zeitzeiger(t(x_train), y_train, t(x_test), sumabsv = 1, nSpc = 2)
results = zzFit['predResult']
predResult = results[["predResult"]]
error = getCircDiff(predResult$timePred, t(y_test))
error = unlist(error, use.names = FALSE)
error = abs(error)
error1 = mean(error)
dfTest = data.frame(timeObs = t(y_test), timePred = predResult$timePred, timeError = getCircDiff(predResult$timePred, t(y_test)))
ggplot(dfTest) +
  geom_point(aes(x = timeObs, y = timePred), size = 2, shape = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed') +
  scale_x_continuous(limits = c(0, 1)) + scale_y_continuous(limits = c(0, 1)) +
  labs(x = 'Observed time', y = 'Predicted time') + theme_bw()
print(error1)


sumabsv = c(1, 1.5, 3)
nSpc = 1:10
nFolds = 6
foldid = sample(rep(1:nFolds, length.out = 12))

fitResultList = zeitzeigerFitCv(t(x_train), y_train, foldid)

spcResultList = list()
for (ii in 1:length(sumabsv)) {
  spcResultList[[ii]] = zeitzeigerSpcCv(fitResultList, sumabsv = sumabsv[ii])}

predResultList = list()
for (ii in 1:length(sumabsv)) {
  predResultList[[ii]] = zeitzeigerPredictCv(t(x_valid), y_valid, foldid,
                                             spcResultList[[ii]], nSpc = nSpc)}

timePredList = lapply(predResultList, function(a) a$timePred)

cvResult = data.frame(do.call(rbind, timePredList),
                      timeObs = rep(y_valid, length(sumabsv)),
                      sumabsv = rep(sumabsv, each = length(time)),
                      obs = rep(1:16, length(sumabsv)),
                      stringsAsFactors = FALSE)

cvResultGath = gather(cvResult, key = nSpc, value = timePred, -obs, -timeObs,
                      -sumabsv)
cvResultGath$nSpc = as.integer(sapply(as.character(cvResultGath$nSpc),
                                      function(a) substr(a, 2, nchar(a))))
cvResultGath$sumabsv = factor(cvResultGath$sumabsv)
cvResultGath$timeError = getCircDiff(cvResultGath$timePred, cvResultGath$timeObs)

cvResultGathGroup = cvResultGath %>%
  group_by(sumabsv, nSpc) %>%
  summarize(medae = median(abs(timeError)))

ggplot(cvResultGathGroup) +
  geom_point(aes(x = nSpc, y = medae, shape = sumabsv, color = sumabsv), size = 2) +
  labs(x = 'Number of SPCs', y = 'Median absolute error') +
  theme_bw() + theme(legend.position = c(0.7, 0.7))


"Make predictions using optimal hyperparameters"





zzFit = zeitzeiger(t(x_train), y_train, t(x_train), sumabsv = 1, nSpc = 10)
results = zzFit['predResult']
predResult = results[["predResult"]]
error = getCircDiff(predResult$timePred, t(y_train))
error = unlist(error, use.names = FALSE)
error = abs(error)
error1 = mean(error)
dfTest = data.frame(timeObs = t(y_train), timePred = predResult$timePred, timeError = getCircDiff(predResult$timePred, t(y_train)))
ggplot(dfTest) +
  geom_point(aes(x = timeObs, y = timePred), size = 2, shape = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed') +
  scale_x_continuous(limits = c(0, 1)) + scale_y_continuous(limits = c(0, 1)) +
  labs(x = 'Observed time', y = 'Predicted time') + theme_bw()
print(error1)

write.csv(dfTest, "Data\\ZeitValidPreds.csv")

zzFit = zeitzeiger(t(x_train), y_train, t(x_valid), sumabsv = 1, nSpc = 10)
results = zzFit['predResult']
predResult = results[["predResult"]]
error = getCircDiff(predResult$timePred, t(y_valid))
error = unlist(error, use.names = FALSE)
error = abs(error)
error1 = mean(error)
dfTest = data.frame(timeObs = t(y_valid), timePred = predResult$timePred, timeError = getCircDiff(predResult$timePred, t(y_valid)))
ggplot(dfTest) +
  geom_point(aes(x = timeObs, y = timePred), size = 2, shape = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed') +
  scale_x_continuous(limits = c(0, 1)) + scale_y_continuous(limits = c(0, 1)) +
  labs(x = 'Observed time', y = 'Predicted time') + theme_bw()
print(error1)

write.csv(dfTest, "Data\\ZeitValidPreds.csv")


zzFit = zeitzeiger(t(x_train), y_train, t(x_test), sumabsv = 1, nSpc = 10)
results = zzFit['predResult']
predResult = results[["predResult"]]
error = getCircDiff(predResult$timePred, t(y_test))
error = unlist(error, use.names = FALSE)
error = abs(error)
error1 = mean(error)
dfTest = data.frame(timeObs = t(y_test), timePred = predResult$timePred, timeError = getCircDiff(predResult$timePred, t(y_test)))
ggplot(dfTest) +
  geom_point(aes(x = timeObs, y = timePred), size = 2, shape = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed') +
  scale_x_continuous(limits = c(0, 1)) + scale_y_continuous(limits = c(0, 1)) +
  labs(x = 'Observed time', y = 'Predicted time') + theme_bw()
print(error1)

write.csv(dfTest, "Data\\ZeitTestPreds.csv")


