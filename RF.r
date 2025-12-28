####***********************************************************************
#### ---- Data preparation ----

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

load("Data/Data_ICEWS_Trim.Rdata")

data_sy <- Data_ICEWS_Trim[, c("ICEWSGOVT_bin", "ICEWSNGOVT_bin", "Year_F", "Mo.x",
                        "ID.x", "ADM1_EN", "ADM2_EN",
                        "GrowingSeason_Month", "TempC", "SPIBelow1sd", 
                        "SPIBelow1.5sd", "Grid_Mo")]

colnames(data_sy) <- c("GOV", "NGOV", "Year", "Month.x", "ID.x", "Dis1", "Dis2",
                       "Grow_season", "TempC", "SPI3_below1", "SPI3_below1.5",
                       "Grid_month")
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
#*Factor variables

data_sy$GOV <- factor(data_sy$GOV, levels = c(0,1), labels = c("no_violence", "violence"))
data_sy$NGOV <- factor(data_sy$NGOV, levels = c(0,1), labels = c("no_violence", "violence"))

data_sy$Grow_season <- as.numeric(data_sy$Grow_season)
data_sy$SPI3_below1 <- as.numeric(data_sy$SPI3_below1)
data_sy$SPI3_below1.5 <- as.numeric (data_sy$SPI3_below1.5)
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
#*Data check

head(data_sy)

levels(data_sy$GOV)
levels(data_sy$NGOV)

range(data_sy$Grow_season)
range(data_sy$SPI3_below1)
range(data_sy$SPI3_below1.5)

range(data_sy$TempC)

nrow(data_sy)

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
#*Prepare data per model
  
  library(dplyr)
  
  data_1 <- data_sy %>%
    select(GOV, Grow_season, SPI3_below1, TempC)
  
  data_2 <- data_sy %>%
    select(NGOV, Grow_season, SPI3_below1, TempC)
  
  data_3 <- data_sy %>%
    select(GOV, Grow_season, SPI3_below1.5, TempC)
  
  data_4 <- data_sy %>%
    select(NGOV, Grow_season, SPI3_below1.5, TempC)
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
  
####***********************************************************************
#### ---- Random Forest ----

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
library(caret)
library(doParallel)

# parallel work

n_cores <- parallel::detectCores()

cl <- makeCluster(n_cores - 1)
registerDoParallel(cl)
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
#*CV 5-fold
#*Settings for ROC/AUC - using model for classification, not prediction

control <- trainControl(method = "cv", number = 5, classProbs = TRUE, 
                        summaryFunction = twoClassSummary,
                        savePredictions = "final",
                        allowParallel = TRUE)
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
#*model 1: GOV violence and moderate dry conditions

RF_model_1 <- train(
  GOV ~ ., 
  data = data_1, 
  method = "rf", 
  metric = "ROC",
  trControl = control, 
  tuneGrid = expand.grid(mtry = c(1,2,3)),      # try 1, 2, 3 random variables for split
  ntree = 500,                                     # make 500 trees
  importance = TRUE
)
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
#*model 2: non GOV violence and moderate dry conditions

RF_model_2 <- train(
  NGOV ~ ., 
  data = data_2, 
  method = "rf", 
  metric = "ROC",
  trControl = control, 
  tuneGrid = expand.grid(mtry = c(1,2,3)),      # try 1, 2, 3 random variables for split
  ntree = 500,                                     # make 500 trees
  importance = TRUE
)
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
#*model 3: GOV violence and severe dry conditions

RF_model_3 <- train(
  GOV ~ ., 
  data = data_3, 
  method = "rf", 
  metric = "ROC",
  trControl = control, 
  tuneGrid = expand.grid(mtry = c(1,2,3)),      # try 1, 2, 3 random variables for split
  ntree = 500,                                     # make 500 trees
  importance = TRUE
)
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
#*model 4: non GOV violence and severe dry conditions

RF_model_4 <- train(
  NGOV ~ ., 
  data = data_4, 
  method = "rf", 
  metric = "ROC",
  trControl = control, 
  tuneGrid = expand.grid(mtry = c(1,2,3)),      # try 1, 2, 3 random variables for split
  ntree = 500,                                     # make 500 trees
  importance = TRUE
)
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
#*Save the models

saveRDS(RF_model_1, "Models/rf_model_1.rds")
saveRDS(RF_model_2, "Models/rf_model_2.rds")
saveRDS(RF_model_3, "Models/rf_model_3.rds")
saveRDS(RF_model_4, "Models/rf_model_4.rds")

#*Print the models // save for separate reporting

print(RF_model_1)
print(RF_model_2)
print(RF_model_3)
print(RF_model_4)

cv_table <- rbind(
  cbind(Model = "Model 1",  RF_model_1$results),
  cbind(Model = "Model 2", RF_model_2$results),
  cbind(Model = "Model 3", RF_model_3$results),
  cbind(Model = "Model 4", RF_model_4$results)
)

cv_table

write.csv(
  cv_table,
  "Final/RF_CV.csv",
  row.names = FALSE
)


####***********************************************************************
#### ---- ROC-AUC Results ----

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

# AUC values saved as auc_table.docx
library(pROC)

auc_table <- data.frame(
  Model = c(
    "Model 1",
    "Model 2",
    "Model 3",
    "Model 4"
  ),
  AUC = c(
    auc(roc(RF_model_1$pred$obs, RF_model_1$pred$violence)),
    auc(roc(RF_model_2$pred$obs, RF_model_2$pred$violence)),
    auc(roc(RF_model_3$pred$obs, RF_model_3$pred$violence)),
    auc(roc(RF_model_4$pred$obs, RF_model_4$pred$violence))
  )
)

auc_table$AUC <- round(auc_table$AUC, 3)

write.csv(auc_table, "Final/auc_tables.csv", row.names = FALSE)

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
# ROC curves

preds1 <- RF_model_1$pred
preds2 <- RF_model_2$pred
preds3 <- RF_model_3$pred
preds4 <- RF_model_4$pred

roc1 <- roc(response = preds1$obs, predictor = preds1$violence)
roc2 <- roc(response = preds2$obs, predictor = preds2$violence)
roc3 <- roc(response = preds3$obs, predictor = preds3$violence)
roc4 <- roc(response = preds4$obs, predictor = preds4$violence)
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
# Plot models and save as RF_ROC.png

png("Final/RF_ROC.png", width = 3000, height = 1000, res = 300)

par(mfrow = c(1,3))

# ---- PANEL 1: models 1 & 2 ----
plot(roc1, col = "maroon", lwd = 2)
plot(roc2, col = "royalblue", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2)

title(main = "(a) Moderate dry conditions\n",
    cex.main = 1.1)
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

#***** ***** ***** ***** ***** ***** ***** ***** ***** *****
# ---- PANEL 2: models 3 & 4 ----
plot(roc3, col = "maroon", lwd = 2)
plot(roc4, col = "royalblue", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2)

title(main = "(b) Severe dry conditions\n",
     cex.main = 1.1)

# ---- PANEL 3: Legend only ----
plot.new()
legend(
  "left",
  legend = c(
    paste("GOV \nAUC (a) =", round(auc(roc1), 2), "\nAUC (b) =", round(auc(roc3), 2),"\n"),
    paste("NGOV \nAUC (a) =", round(auc(roc2), 2), "\nAUC (b) =", round(auc(roc4), 2),"")
  ),
  col = c("maroon", "royalblue"),
  lwd = 2,
  bty = "n",
  cex = 1.1
)

dev.off()

# reset plotting defaults
par(mfrow = c(1,1))
#***** ***** ***** ***** ***** ***** ***** ***** ***** *****

####***********************************************************************
#### ---- Feature importance ----

# nice names
label_map <- c(
  TempC         = "Temperature (°C)",
  Grow_season   = "Growing season",
  SPI3_below1   = "Moderate dry conditions",
  SPI3_below1.5 = "Severe dry conditions"
)

# function to rename variables
rename_vi_labels <- function(vi, label_map) {
  rn <- rownames(vi$importance)
  
  common <- intersect(rn, names(label_map))
  rn[rn %in% common] <- label_map[common]
  
  rownames(vi$importance) <- rn
  vi
}

# feature importance in figure
png("Final/RF_importance.png", width = 2400, height = 2400, res = 300)

#par(mfrow = c(2, 2))

library(gridExtra)

plot_vi <- function(model, title, label_map) {
  vi <- varImp(model)
  
  rn <- rownames(vi$importance)
  rn[rn %in% names(label_map)] <- label_map[rn[rn %in% names(label_map)]]
  rownames(vi$importance) <- rn
  
  plot(vi, main = title)
}

p1 <- plot_vi(RF_model_1, "Model 1", label_map)
p2 <- plot_vi(RF_model_2, "Model 2", label_map)
p3 <- plot_vi(RF_model_3, "Model 3", label_map)
p4 <- plot_vi(RF_model_4, "Model 4", label_map)

grid.arrange(p1, p2, p3, p4, ncol = 2)

dev.off()
