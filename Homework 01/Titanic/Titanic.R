if (!'dplyr' %in% rownames(installed.packages())) {install.packages('dplyr')}
library('dplyr')
if (!'ggplot2' %in% rownames(installed.packages())) {install.packages('ggplot2')}
library('ggplot2')
if (!'scales' %in% rownames(installed.packages())) {install.packages('scales')}
library('scales')
if (!'mice' %in% rownames(installed.packages())) {install.packages('mice')}
library('mice')
if (!'randomForest' %in% rownames(installed.packages())) {install.packages('randomForest')}
library('randomForest')


# Getting data
train <- read.csv('../input/train.csv')
test <- read.csv('../input/test.csv')

full <- bind_rows(train, test)
str(full)


# Feature engineering I
## Get passenger title
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

table(full$Sex, full$Title)
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'
table(full$Sex, full$Title)

## Do families sink or swim together?
full$FSize <- full$SibSp + full$Parch + 1
plot_FSize_Survived <- ggplot(full[1:nrow(train), ], aes(x = FSize, fill = factor(Survived)))
plot_FSize_Survived <- plot_FSize_Survived +
    geom_bar(stat = 'Count', position = 'dodge') +
    scale_x_continuous(breaks = c(1:11)) +
    labs(x = "Family Size")
print(plot_FSize_Survived)

full$FSizeD[full$FSize == 1] <- 'singleton'
full$FSizeD[1 < full$FSize & full$FSize < 5] <- 'small'
full$FSizeD[4 < full$FSize] <- 'large'
mosaicplot(table(full$FSizeD, full$Survived), main = 'Family size by survival', shade = TRUE)

## Deck
full$Deck <- factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))


# Missingness
## Embarked
table(full$Embarked)
missingEmbarked <- full[full$Embarked == "", "PassengerId"]

embarkFare <- full %>%
    filter(!Embarked == "")
plot_Embark_Fare <- ggplot(embarkFare, aes(x = Embarked, y = Fare, fill = factor(Pclass)))
plot_Embark_Fare <- plot_Embark_Fare +
    geom_boxplot() +
    geom_hline(aes(yintercept = full[missingEmbarked[1], "Fare"]), colour = 'red', linetype = 'dashed', lwd = 2) +
    scale_y_continuous(labels = dollar_format())
print(plot_Embark_Fare)

full$Embarked[missingEmbarked] <- 'C'

## Fare
summary(full$Fare)
missingFare <- full[is.na(full$Fare), "PassengerId"]
plot_Fare <- ggplot(full[full$Pclass == full[missingFare[1], "Pclass"] &
                             full$Embarked == full[missingFare[1], "Embarked"], ],
                    aes(x = Fare))
plot_Fare <- plot_Fare +
    geom_density(fill = '#99d6ff', alpha = 0.4) +
    geom_vline(aes(xintercept = median(Fare, na.rm = TRUE)), colour = 'red', linetype = 'dashed', lwd = 1) +
    scale_x_continuous(labels = dollar_format())
print(plot_Fare)

full$Fare[missingFare] <- median(full[full$Pclass == full[missingFare[1], "Pclass"] &
                                          full$Embarked == full[missingFare[1], "Embarked"], "Fare"],
                                 na.rm = TRUE)

## Predictive imputation
summary(full$Age)

full$Surname <- sapply(full$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])
full$Family <- paste(full$Surname, full$FSize, sep = '_')
factor_vars <- c('PassengerId', 'Pclass', 'Sex', 'Embarked', 'Title', 'Surname', 'Family', 'FSizeD')
full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

set.seed(9231066)
mice_mod <- mice(full[, !names(full) %in%
                          c('PassengerId', 'Name', 'Ticket', 'Cabin', 'Family', 'Surname', 'Survived')],
                 method = 'rf')
mice_output <- complete(mice_mod)

par(mfrow = c(1, 2))
hist(full$Age, freq = F, main = 'Age: Original Data', col = 'darkgreen', ylim = c(0, 0.04))
hist(mice_output$Age, freq = F, main = 'Age: MICE Output', col = 'lightgreen', ylim = c(0, 0.04))

full$Age <- mice_output$Age
summary(full$Age)

# Feature engineering II
## Child and mother features
plot_Age_Survived_Sex <- ggplot(full[1:891, ], aes(Age, fill = factor(Survived)))
plot_Age_Survived_Sex <- plot_Age_Survived_Sex +
    geom_histogram() +
    facet_grid(.~Sex)
print(plot_Age_Survived_Sex)

full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'
table(full$Child, full$Survived)

full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'
table(full$Mother, full$Survived)

full$Child <- as.factor(full$Child)
full$Mother <- as.factor(full$Mother)

# Prediction
train <- full[1:nrow(train), ]
test <- full[(nrow(train) + 1):(nrow(train) + nrow(test)), ]

## Random Forest model
set.seed(9231066 + 2)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title +
                             FSizeD + Child + Mother,
                         data = train)
par(mfrow = c(1, 1))
plot(rf_model, ylim = c(0, 0.36))
legend('topright', colnames(rf_model$err.rate), col = 1:3, fill = 1:3)

## Variable importance
importance <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance),
                            Importance = round(importance[, 'MeanDecreaseGini'], 2))
rankImportance <- varImportance %>%
    mutate(Rank = paste0('#', dense_rank(desc(Importance))))
plot_Importance <- ggplot(rankImportance,
                          aes(x = reorder(Variables, Importance), y = Importance, fill = Importance))
plot_Importance <- plot_Importance +
    geom_bar(stat = 'identity') +
    geom_text(aes(Variables, y = 0.5, label = Rank), hjust = 0, vjust = 0.55, size = 4, colour = 'red') +
    labs(x = 'Variables') +
    coord_flip()
print(plot_Importance)

## Output
prediction <- predict(rf_model, test)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)
write.csv(solution, file = 'rf_mod_solution.csv', row.names = FALSE)
