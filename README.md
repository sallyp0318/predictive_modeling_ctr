## Introduction

The objective for this competition was to develop a robust predictive model that minimizes RMSE (Root Mean Squared Error) when providing click through rate (CTR) predictions from the advertising metrics by ensuring the predictions generalize well to unseen data. Throughout this analysis, various challenges were addressed, and multiple modeling strategies were applied. Personally, the metrics were familiar as my professional background focuses on advertising. However, selecting and engineering features was still a challenge when planning the analysis since majority of the metrics were those that significant impact and relevant to CTR levels in real life.

## Preparation

Working directory was set. Relevant libraries were loaded, as well as the data needed (analysis and scoring).

```{r  echo=T, results='hide'}
# Load necessary libraries
library(dplyr)
library(tidyr)
library(caret)
library(randomForest)
library(xgboost)
library(Metrics)
library(tibble)
library(rpart)
library(ggplot2)

# Load the data
analysis_data <- read.csv('/Users/sallypark/Desktop/Columbia/Fall 2024/5200 Framework and Methods I/Data/CTR Predictions/analysis_data.csv')
scoring_data <- read.csv('/Users/sallypark/Desktop/Columbia/Fall 2024/5200 Framework and Methods I/Data/CTR Predictions/scoring_data.csv')
```

## Data Cleaning

Prior to modeling, CTR was separated out from the analysis data set and was added back for modeling for a smoother process of preparing the data.

```{r }
# Separate the target variable
target <- analysis_data$CTR
analysis_data <- analysis_data |> select(-CTR) 
```

Dimension and structure of the data sets were first checked. Analysis data had 4000 observations and 29 variables, including CTR. Scoring data had 1000 observations and missing CTR which is the target variable that needs to be predicted and added at the end of the modeling.

```{r }
# Check data structure 
str(analysis_data)
str(scoring_data)
```

```{r }
# Check data dimensions 
dim(analysis_data)
dim(scoring_data)
```

After checking the dimension and structure of variables, first observation during the data exploration was the number of missing values. The missing values were identified across both numerical and categorical features. The categorical values that were missing were gender, age_group, and location.

```{r }
# Check for missing values
colSums(is.na(analysis_data))
```

For numerical features, mean imputation was employed, while for categorical variables, mode imputation function was created and used to handle the missing values effectively. This ensured that the missing values did not introduce bias or affect the performance of the models.

```{r }
# Impute missing values in numeric columns with mean, categorical with mode
get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Apply imputation
analysis_data <- analysis_data |>
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .))) |>
  mutate(across(where(is.character), ~ ifelse(is.na(.), get_mode(.), .)))

scoring_data <- scoring_data |>
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .))) |>
  mutate(across(where(is.character), ~ ifelse(is.na(.), get_mode(.), .)))
```

For other submissions during this process, KNN and predictive imputation were also considered and ran, but the predictions from the imputation methods did not drive a lower RMSE. Due to the lower performance and to maintain simplicity and scalability, KNN imputation was avoided. A decision tree based imputation was also considered for missing categorical values, in an attempt to use patterns in other variables when imputing. With the decision tree based imputation, my thought process was that there may be a more context-aware method than simply imputing with modes. However, RMSE with decision tree based imputation also did not show substantial improvement in predictions.

After categorical features were imputed, they were converted to factors. All categorical columns were added to a vector separately and converted to factors for a faster converting process at once. It was applied across both analysis and scoring data sets.

```{r }
# Convert categorical columns to factors
categorical_cols <- c("position_on_page", "ad_format", "age_group", "gender", "location", "time_of_day", "day_of_week", "device_type")
analysis_data[categorical_cols] <- lapply(analysis_data[categorical_cols], as.factor)
scoring_data[categorical_cols] <- lapply(scoring_data[categorical_cols], as.factor)
```

Then, the CTR was added back for correlation analysis.

```{r }
# Add CTR back to analysis_data for correlation analysis
analysis_data$CTR <- target
```

I wanted to look at how the CTR values were distributed. It is compromised with values mostly near zero and significantly skewed. With prior domain knowledge about advertising ecosystem and metrics, these values were reflecting real-life scenario.

```{r , echo=FALSE}
# Check distribution of CTR
ggplot(analysis_data, aes(x = CTR)) +
  geom_histogram(binwidth = 0.001, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of CTR",
       x = "CTR",
       y = "Frequency") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14)
  )
```

While this was not considered in any of the submissions, scaling the CTR levels may have benefited the predictive analysis. However, I believe that the feature engineering helped mitigate this missed step. The distribution also shows the need for feature engineering as it strongly suggest there to be more of a non-linear relationship, which was explored with other types of models than linear regression.

The relationship between the predictors and the target were initially analyzed using correlation metrics. When investigating variables that are highly correlated with CTR, those with absolute value of correlation higher than 0.5 and suggested their potential predictive power were considered.

```{r ,echo=FALSE}
# Calculate correlations with CTR
correlations <- analysis_data |>
  select_if(is.numeric) |>  # Select only numeric columns
  select(-CTR) |>  # Exclude CTR itself to avoid self-correlation
  summarise(across(everything(), ~ cor(.x, analysis_data$CTR, use = "complete.obs"))) |>
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Correlation") |>
  arrange(desc(abs(Correlation)))

# Plot correlations as a bar graph
ggplot(correlations, aes(x = reorder(Variable, -abs(Correlation)), y = Correlation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(Correlation, 2)), vjust = -0.5) +
  theme_minimal() +
  labs(title = "Correlation of Variables with CTR",
       x = "Variable",
       y = "Correlation Coefficient") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

```{r }
# Calculate correlations and select variables highly correlated with CTR (e.g., abs(cor) > 0.5)
correlations <- cor(analysis_data |> select(where(is.numeric)), use = "complete.obs")
high_corr_vars <- names(which((correlations["CTR", ] > 0.5 & correlations["CTR", ] <= 1) | 
                                (correlations["CTR", ] < -0.5 & correlations["CTR", ] >= -1)))
high_corr_vars <- setdiff(high_corr_vars, "CTR")  # Remove CTR itself
high_corr_vars
```

‘visual_appeal’ was the only variable that met this criteria. Visualization was created for an easier interpretation of the correlations, and the high_corr_vars objective was created for storage. Data sets were created for both analysis and scoring that only had variables with high correlation for modeling.

```{r }
# Create data sets for high-correlation variables
analysis_data_high_corr <- analysis_data |> select(all_of(high_corr_vars))
scoring_data_high_corr <- scoring_data |> select(all_of(high_corr_vars))
analysis_data_high_corr$CTR <- target  # Add CTR back for modeling
```

## Feature Engineering

Feature engineering also played a significant role in enhancing predictive performance. Relevance_score and ad_strength were added as variables by aggregating related features (targeting_score and contextual_relevance for relevance_score, and visual_appeal and cta_strength for ad_strength). Difference in text length was also added where the body_text_length is subtracted from the headline_length.

```{r }
# Add engineered features
analysis_data <- analysis_data |>
  mutate(
    relevance_score = targeting_score + contextual_relevance,      
    ad_strength = visual_appeal + cta_strength,                    
    text_length_diff = headline_length - body_text_length          
  )

scoring_data <- scoring_data |>
  mutate(
    relevance_score = targeting_score + contextual_relevance,
    ad_strength = visual_appeal + cta_strength,
    text_length_diff = headline_length - body_text_length
  )
```

Selected variables were than grouped separately in a vector form. This objective compromised of predictors with the highest potential. 'visual_appeal' was added as it was the highest correlated metric with CTR. Variables such as targeting_score and contextual_relevance were also selected due to their relevant domain impact. Similarly, others were added as they represent content characteristics or user engagement factors. As CTR is driven by user interaction, content relevance, and ad formatting, the other variables were that with the biggest predictive power by context or domain knowledge.

```{r }
# Selected engineered features
selected_vars <- c("visual_appeal", "targeting_score", "headline_length", 
                   "cta_strength", "body_word_count","headline_word_count",
                   "body_text_length", "headline_sentiment", 
                   "contextual_relevance", "relevance_score", "ad_strength", "text_length_diff")
```

In an attempt to improve RMSE, another submission had more feature engineering added, such as body character per word from word count and text length variables and engagement potential with ad frequency and market saturation. However, it ultimately did not add value to predictive performance. This may be due to overfitting, as the complex and unnecessary number of feature engineering and parameter turning could potentially increase the risk, especially on small training data sets.

Engineered and selected variables were then separately added to a vector form. Data sets were created for analysis and scoring that only comprised of those predictors.

```{r }
# Create separate datasets for selected variables and full variables
analysis_data_selected <- analysis_data |> select(all_of(selected_vars))
scoring_data_selected <- scoring_data |> select(all_of(selected_vars))
```

## Encoding and Scaling

Factor levels were aligned between scoring and analysis to avoid inconsistencies. To facilitate model training, a one-hot encoding ‘dummyVars’ function from the caret package was used to ensure that the models could process the data effectively. Encoding was performance across both analysis and scoring data sets to avoid feature mismatches during the prediction. This was added due to errors that I have ran into when running the models that suggested there to be inconsistencies between the two data, and have helped mitigate last minute changes to the code, after the modeling was running for a longer amount of time. Therefore, this was done across all data sets that were utilized for modeling, which included data sets with all variables, variable with high correlation, or visual_appeal, and selected variables with the highest predictive power.

```{r  }
# Remove CTR from analysis_data before creating dummy variables
analysis_data_no_ctr <- analysis_data |> select(-CTR)

# Ensure factor levels are consistent between analysis_data and scoring_data
for (col in categorical_cols) {
  levels(scoring_data[[col]]) <- levels(analysis_data[[col]])
}

# Full-Variable Data Encoding
dummy_full <- dummyVars("~ .", data = analysis_data_no_ctr)
analysis_data_full_encoded <- predict(dummy_full, newdata = analysis_data_no_ctr) |> as.data.frame()
scoring_data_full_encoded <- predict(dummy_full, newdata = scoring_data) |> as.data.frame()

# High Correlation Variable Data Encoding
dummy_high_corr <- dummyVars("~ .", data = analysis_data_high_corr |> select(-CTR))
analysis_data_high_corr_encoded <- predict(dummy_high_corr, newdata = analysis_data_high_corr |> 
                                             select(-CTR)) |> as.data.frame()
scoring_data_high_corr_encoded <- predict(dummy_high_corr, newdata = scoring_data_high_corr) |> as.data.frame()

# Selected Variable Data Encoding
dummy_selected <- dummyVars("~ .", data = analysis_data_selected)
analysis_data_selected_encoded <- predict(dummy_selected, newdata = analysis_data_selected) |> as.data.frame()
scoring_data_selected_encoded <- predict(dummy_selected, newdata = scoring_data_selected) |> as.data.frame()

# Add CTR back to the encoded analysis_data for modeling
analysis_data_full_encoded$CTR <- target
analysis_data_high_corr_encoded$CTR <- target
analysis_data_selected_encoded$CTR <- target
```

## Data Splitting

After encoding, the analysis data set was split into training and testing subsets with the createDataPartition function. The data set was split into training (80%) and testing (20%) subsets to evaluate model performance on unseen data. As there were 3 data sets with different predictors were used for different models, each data sets was split into training and testing subsets. For simplicity, seed 123 was set.

```{r }
set.seed(123)
trainIndex_full <- createDataPartition(analysis_data_full_encoded$CTR, p = 0.8, list = FALSE)

# Data Splitting for Full, Selected, and High Correlation Datasets

# Full-variable data split
train_data_full <- analysis_data_full_encoded[trainIndex_full, ]
test_data_full <- analysis_data_full_encoded[-trainIndex_full, ]

# Selected-variable data split
set.seed(123)
trainIndex_selected <- createDataPartition(analysis_data_selected_encoded$CTR, p = 0.8, list = FALSE)
train_data_selected <- analysis_data_selected_encoded[trainIndex_selected, ]
test_data_selected <- analysis_data_selected_encoded[-trainIndex_selected, ]

# High-correlation-variable data split
set.seed(123)
trainIndex_high_corr <- createDataPartition(analysis_data_high_corr_encoded$CTR, p = 0.8, list = FALSE)
train_data_high_corr <- analysis_data_high_corr_encoded[trainIndex_high_corr, ]
test_data_high_corr <- analysis_data_high_corr_encoded[-trainIndex_high_corr, ]
```

## Model Training and Evaluation

fitControl object with trainControl function was used to ensure proper validation and training models. It specifies the k-fold cross-validation, indicating the data set split into 5 folds. This helped having the codes more organized especially for Random Forest and XGBoost models which were directly impacted by the setting. This was specifically added after optimizing hyperparameters for XGBoost in other submissions, as they resulted in overfitting and worse RMSE.

```{r }
# hyperparameter objective
fitControl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
```

As the objective of this analysis focused on Root Mean Squared Error, the RMSE evaluation function was defined and implemented for all model results for a simpler and faster comparison across the models.

```{r }
# Define RMSE evaluation function
rmse_eval <- function(model, data, target) {
  predictions <- predict(model, newdata = data)
  rmse(target, predictions)
}
```

Several models were explored to predict CTR, including linear regression, decision trees, random forests, and XGBoost. Decision tree models with full variables continued to run into errors and ultimately were avoided. This may be due to overfitting since the data set is relatively small. The error however was mostly due to observed missing data from the train data set. After confirming there was no missing data, I was not able to solve or utilize the model. From prior submissions, only using the highly correlated variable (visual_appeal) for linear regression and decision tree drove the worst RMSE and was not run for comparison.

As mentioned, hyper-parameters were optimized separately in another submission. 'max_depth', 'eta' (learning rate), 'subsample', and 'min_child_weight' were tuned using grid search and cross-validation. Regularization parameters like 'lambda' and 'alpha' were also adjusted to prevent overfitting. While RMSE was lower, the current submission gave a better public score in the leaderboard that looked at unseen data, suggesting that the default parameters were better to use compared to the optimized parameters which may have been adjusted too aggressively. Overfitting became apparent when models achieved low RMSE on training data but performed poorly on public leaderboard.

```{r  echo=T, results='hide'}
# Model Training for Each Dataset (Full, Selected, and High Correlation Variables)
# 1. Full Variables - Decision Tree (deleted)
#set.seed(123)
#dt_full <- train(CTR ~ ., data = train_data_full, method = "rpart", trControl = fitControl, metric = "RMSE")
#dt_rmse_full <- rmse_eval(dt_full, test_data_full, test_data_full$CTR)

# 2. Full Variables - Random Forest
set.seed(123)
rf_full <- train(CTR ~ ., data = train_data_full, method = "rf", trControl = fitControl, metric = "RMSE")
rf_rmse_full <- rmse_eval(rf_full, test_data_full, test_data_full$CTR)

# 3. Full Variables - XGBoost
set.seed(123)
xgb_full <- train(CTR ~ ., data = train_data_full, method = "xgbTree", trControl = fitControl, metric = "RMSE")
xgb_rmse_full <- rmse_eval(xgb_full, test_data_full, test_data_full$CTR)

# 4. Full Variables - Linear Regression
set.seed(123)
lm_full <- train(CTR ~ ., data = train_data_full, method = "lm", trControl = fitControl, metric = "RMSE")
lm_rmse_full <- rmse_eval(lm_full, test_data_full, test_data_full$CTR)

# 5. Selected Variables - Decision Tree
set.seed(123)
dt_selected <- train(CTR ~ ., data = train_data_selected, method = "rpart", trControl = fitControl, metric = "RMSE")
dt_rmse_selected <- rmse_eval(dt_selected, test_data_selected, test_data_selected$CTR)

# 6. Selected Variables - Random Forest
set.seed(123)
rf_selected <- train(CTR ~ ., data = train_data_selected, method = "rf", trControl = fitControl, metric = "RMSE")
rf_rmse_selected <- rmse_eval(rf_selected, test_data_selected, test_data_selected$CTR)

# 7. Selected Variables - XGBoost
set.seed(123)
xgb_selected <- train(CTR ~ ., data = train_data_selected, method = "xgbTree", trControl = fitControl, metric = "RMSE")
xgb_rmse_selected <- rmse_eval(xgb_selected, test_data_selected, test_data_selected$CTR)

# 8. Selected Variables - Linear Regression
set.seed(123)
lm_selected <- train(CTR ~ ., data = train_data_selected, method = "lm", trControl = fitControl, metric = "RMSE")
lm_rmse_selected <- rmse_eval(lm_selected, test_data_selected, test_data_selected$CTR)

# 9. High Correlation Variables - Random Forest
set.seed(123)
rf_high_corr <- train(CTR ~ ., data = train_data_high_corr, method = "rf", trControl = fitControl, metric = "RMSE")
rf_rmse_high_corr <- rmse_eval(rf_high_corr, test_data_high_corr, test_data_high_corr$CTR)

# 10. High Correlation Variables - XGBoost
set.seed(123)
xgb_high_corr <- train(CTR ~ ., data = train_data_high_corr, method = "xgbTree", trControl = fitControl, metric = "RMSE")
xgb_rmse_high_corr <- rmse_eval(xgb_high_corr, test_data_high_corr, test_data_high_corr$CTR)
```

Between the models, the trade offs between model complexity and generalization was apparent. This was expected as the CTR values and metrics shows potential non-linear relationship at the start of the analysis. Simpler models like linear regression offered interpretability but lacked flexibility to capture non-linear relationships which was shown by giving the highest RMSE consistently. On the other hand, complex models like XGBoost with default hyperparameters and engineered features produced the best results on both the training data and the public leaderboard. While XGBoost usually calls for tuning and optimization to avoid overfitting, default settings were used as mentioned.

Another version with Support Vector Machines (SVM) modeling technique was conducted to explore further non-linear relationships with an effort to avoid overfitting to noise. However, this also did not beat the RMSE achieved from XGBoost. All submissions used 5-fold cross-validation with trainControl. The latter versions of the submissions also had tuneGrid for hyper-parameter optimization in the SVM and XGBoost models which was ultimately deleted after seeing lower performance.

## Model Comparison

RMSE evaluation was simplified across the board to easily compare the RMSE from all different models and streamline evaluation. Separate codes were added to identify the best model based on the lowest RMSE.

```{r }
# Create a data frame to store RMSE results for all models
rmse_results_combined <- data.frame(
  Model = c("Random Forest - Full Vars", "XGBoost - Full Vars", "Linear Regression - Full Vars",
            "Decision Tree - Selected Vars", "Random Forest - Selected Vars", "XGBoost - Selected Vars", "Linear Regression - Selected Vars","Random Forest - High Corr Vars", "XGBoost - High Corr Vars"),
  RMSE = c(rf_rmse_full, xgb_rmse_full, lm_rmse_full,
           dt_rmse_selected, rf_rmse_selected, xgb_rmse_selected, lm_rmse_selected,
           rf_rmse_high_corr, xgb_rmse_high_corr)
); rmse_results_combined
```

In general, the models with the selected and engineered features had the lowest RMSE compared to those with full variables or high correlated variable (visual appeal). Out of all modeling techniques, XGBoost consistently was the best performing for all 3 types of set of variables.

```{r }
# Identify the best model based on the lowest RMSE
best_rmse <- min(rmse_results_combined$RMSE)
best_model_name <- rmse_results_combined$Model[which.min(rmse_results_combined$RMSE)]

# Display the best model and its RMSE
cat("Best Model:", best_model_name, "\nRMSE:", best_rmse)
```

Predictions data frame was generated with the scoring data set using the best model with the lowest RMSE and was prepared for the submission data frame.

```{r }
# Load prediction data (e.g., scoring_data)
prediction_data <- scoring_data
```

To automate the process of the model to be used for predictions, if and else if functions were used to identify the best model, in this case was "XGBoost - Selected Vars".

```{r }
# Select the best model based on the name
if (best_model_name == "Random Forest - Full Vars") {
  best_model <- rf_full
} else if (best_model_name == "XGBoost - Full Vars") {
  best_model <- xgb_full
} else if (best_model_name == "Linear Regression - Full Vars") {
  best_model <- lm_full
} else if (best_model_name == "Decision Tree - Selected Vars") {
  best_model <- dt_selected
} else if (best_model_name == "Random Forest - Selected Vars") {
  best_model <- rf_selected
} else if (best_model_name == "XGBoost - Selected Vars") {
  best_model <- xgb_selected
} else if (best_model_name == "Linear Regression - Selected Vars") {
  best_model <- lm_selected
} else if (best_model_name == "Random Forest - High Corr Vars") {
  best_model <- rf_high_corr
} else if (best_model_name == "XGBoost - High Corr Vars") {
  best_model <- xgb_high_corr
}
```

Then the submission file was created with just IDs and CTR predictions. This was the sixth attempt of the entire submission process (12 submissions), which compromised of different formats for modeling codes, feature engineering, correlation analysis, and more variations of hyper-parameters for XGBoost which was mentioned previously.

```{r }
# Generate predictions using the best model
predictions <- predict(best_model, newdata = prediction_data)

# Prepare submission data frame
submission <- data.frame(ID = scoring_data$id, CTR = predictions)

# Write submission to CSV file
write.csv(submission, "CTR_predictions.csv", row.names = FALSE)
```

## Conclusion

After conducting this analysis, I believe that the domain knowledge I carried supported a lot of time management in understanding the data set. To create further feature engineering that created big impact to model performance and enable adding meaningful variables, knowledge on the domain was critical. As I have dealt with a lot of engagement metrics and ad-specific variables, it was a smooth process to understand the variables and its impact volume.

Second was the validation strategies, such as cross-validation and testing to ensure that the models generalized well to unseen data. The more specific and limited the hyper-parameters were, the chance of overfitting grew exponentially. This needed more careful and required further statistical knowledge to be efficient as the model had to run for a number of hours and evaluating no improvement in RMSE due to the overfitting that occurred.

Lastly, data preparation was also an important step in driving improvements, as there were a number of ways to impute missing data and ensure that both data sets were reflecting the same methodology and matching variables. I believe that if there were more data points, and for real-life scenarios, mode imputation for categorical variables may not be enough to achieve acceptable predictions.

In conclusion, this project demonstrated the power of machine learning techniques in addressing real-world prediction tasks. While the final results were influenced by both the modeling approach and the inherent characteristics of the data, the process itself highlighted the importance of preparation, experimentation, and adaptability in achieving analytical success. I believe I was able to experience machine learning and conducting predictive analysis hands-on which was an insightful process, and I hope to be exposed to better processes and apply techniques for further predictions.
