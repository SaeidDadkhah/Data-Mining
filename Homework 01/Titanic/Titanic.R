if (!'dplyr' %in% rownames(installed.packages())) {install.packages('dplyr')}
library('dplyr')
if (!'ggplot2' %in% rownames(installed.packages())) {install.packages('ggplot2')}
library('ggplot2')
if (!"scales" %in% rownames(installed.packages())) {install.packages('scales')}
library('scales')

# Getting data
train <- read.csv('../train.csv')
test <- read.csv('../test.csv')

full <- bind_rows(train, test)
str(full)

# Feature engineering
## Get passenger title
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

table(full$Name, full$Title)
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
