---
layout: post
title: Milestone 2
---

Authors: Anissa Djouhri, Qianyun Shen, Gafran Ijaz, Saleem Ibrahim (Team 1)
## <span style="color:#1E90FF;">Feature Engineering I Question</span>
code related to it: ift6758/data/feature_engineering_one.py ift6758/notebook/feature_engineering_one.ipynb



Using the functionality we developed in **Milestone 1**, we introduced several new features to better describe each shot event:

- **distance_to_net** – the distance between the shot (or goal) and the net.
- **shot_angle** – the angle formed between the shooter and the goal line.
- **is_goal** – indicates whether the shot resulted in a goal (`1`) or not (`0`).
- **is_empty_net** – identifies whether the shot or goal occurred when the net was empty.

We then analyzed how these  features influence the likelihood of scoring by visualizing the relationships between **distance**, **angle**, and **goal outcomes**.



###  Shot Counts by Distance (Goals vs No-Goals)

![Shot Counts by Distance](public/milestone2/image/FeatureEngineering1/distanceShotCounts.png)
<br>
The first histogram shows that most shots are taken within **0–65 feet** from the net.
As distance increases, both the total number of shots and the number of goals decline sharply.
This trend highlights how **proximity to the net** is a major factor in shot success — closer shots are not only more frequent but also far more effective.



### Shot Counts by Angle (Goals vs No-Goals)
 ![Shot Counts by Angle](public/milestone2/image/FeatureEngineering1/angleShotCounts.png)
 <br>
The second plot examines the effect of **shot angle**.
Shots taken from **smaller angles** (directly in front of the net) are much more common and substantially more successful.
As the angle widens (shots from the side), both shot frequency and goal proportion decrease — reinforcing that players positioned in front of the net have a clear advantage.



###  2D Histogram — Distance vs Angle
 ![2D Histogram: Distance vs Angle](public/milestone2/image/FeatureEngineering1/shot_distribution.png)
 <br>
The 2D histogram highlights the **joint distribution** of distance and angle.
The **color intensity** represents shot frequency:
- **Brighter colors** (e.g., *yellow*, *light blue*) → more frequent shot zones.
- **Darker colors** (e.g., *pink*, *purple*) → less frequent shot zones.

Most shots cluster in the **lower-left region** (short distance and small angle), showing that players tend to shoot from **high-probability areas near the center front of the net**.




To further explore the relationship between shot characteristics and scoring success, we examined the **goal rate** of both **distance to the net** and **angle to the net**.


### 1. Goal Rate vs. Distance to Net
 ![Goal Rate vs Distance](public/milestone2/image/FeatureEngineering1/goalRatevsDistance.png)
 <br>
From the first figure, we observe that the **goal rate is highest when players are close to the net**, confirming that short-range shots are the most effective.
Interestingly, the goal rate shows a **small rebound at distances greater than 100 feet**, which helps explain the few long-distance shots observed earlier in the *Shot Counts by Distance* figure.
These long-range goals typically correspond to **empty-net situations**, where the opposing team has pulled its goalie.



### 2. Goal Rate vs. Angle to Net
![Goal Rate vs Angle](public/milestone2/image/FeatureEngineering1/goalRatevsAngle.png)
<br>
In the second plot, the **goal rate decreases as the shot angle increases** — shots taken from the sides of the rink are much less likely to score compared to those from central positions.
The **highest success rates occur at angles near 0°**, directly in front of the net, while shots taken at **10–20° or beyond 60–70°** show very low scoring probabilities.

### Empty Net vs. Non-Empty Net Goals
![Goal Distances: Empty Net vs Non-Empty Net](public/milestone2/image/FeatureEngineering1/goalDistanceHistogram.png)
<br>
To check to see if our data makes sense, we compared goal distances between **empty-net** and **non-empty-net** situations.
The histogram shows that most non-empty-net goals happen within **0–40 feet** of the net — exactly what we’d expect in real games, where goals usually come from close range when a goalie is defending. Almost all goals fall within **60 feet**, comfortably inside the offensive zone.

We didn’t find any major anomalies in the data. There were no non-empty-net goals scored from extremely long distances (over **100 feet**), which could have suggested mislabeled events or swapped coordinates.

By contrast, **empty-net goals** are very uncommon and tend to be scored from **much farther away**, often beyond **70 feet**. This makes sense because empty-net situations occur when the opposing team pulls their goalie for an extra attacker, leaving the net unguarded and giving players the chance to score from long range.


## <span style="color:#1E90FF;">Baseline Models Question</span>

code related to it: notebooks/simple_model.ipynb

![LogisticRegressionAccuracy](public/milestone2/image/baselineModels/logisticRegressionAccuracy.png)
<br>
The logistic regression model trained using **only distance_to_net** achieved 90.5% validation accuracy. However, a closer inspection revealed that the model predicted **no goals** for all shots.

This is an example of accuracy being misleading in highly imbalanced datasets. Only ~9% of shots are goals in the validation set, so predicting "no goal" for every shot yields similar accuracy.

The main issues are: the severe class **imbalance**, the limited feature set, and the default prediction threshold. To improve the model, we should add more predictive features and use evaluation metrics that account for imbalance, such as precision, recall, F1-score, or ROC-AUC.





To quantify the predictive ability of our models, we computed the **Area Under the ROC Curve (AUC)** for each of the four models:

![ROCCurveComparison](public/milestone2/image/baselineModels/ROCCurveComparison.png)

| **Model** | **Features Used**              | **AUC** |
|:-----------|:------------------------------|:-------:|
| Model 1    | Distance to Net               | **0.701** |
| Model 2    | Shot Angle                    | **0.508** |
| Model 3    | Distance + Angle              | **0.701** |
| Model 4    | Random Baseline               | **0.495** |





As shown in the above figure, the **random classifier** forms a perfect diagonal line, which represents random guessing.
By contrast, our models clearly perform better:
the one trained on both **distance** and **angle** features and only **distance** feature achieve the best ROC curve, reaching an AUC of approximately **0.70**, indicating a solid ability to separate goals from non-goals.
while the one trained only on angle contributes little additional discriminative power.
This result confirms that distance remains the dominant factor influencing goal probability.

![GoalRateShotProbability](public/milestone2/image/baselineModels/GoalRateShotProbability.png)

The **Goal Rate vs. Shot Probability Percentile** plot evaluates how well each model ranks shots by their likelihood of resulting in a goal.
The x-axis represents shot probability percentiles  while the y-axis shows the actual goal rate (%) for shots within each percentile group.


As shown in figure above, the **random baseline** (red curve) remains roughly flat, which is expected since it assigns probabilities randomly.
In contrast, the models trained on real features show a clear separation in performance.
The **combined distance + angle model** (green curve)  and  **distance-only model** (blue) consistently achieves the highest goal rate among all models, particularly at the top percentiles, indicating that it successfully identifies high-probability scoring chances.
However, the **angle-only model** (orange) fluctuates more and stays closer to the random baseline, demonstrating less stability and predictive power.
Overall, this figure confirms that incorporating both **distance** and **angle** enhances the model’s ability to rank shots by their true likelihood of becoming goals.


![CumulativeGoalProportion](public/milestone2/image/baselineModels/CumulativeGoalProportion.png)

The **Cumulative Goal Proportion** plot illustrates how goals accumulate across increasing model-predicted probability percentiles.
An ideal model should have a steep curve at the beginning, meaning that most of the real goals fall into the top probability percentiles predicted by the model.

As shown in the figure above, the **random baseline** (red curve) increases linearly — as expected, since it assigns random probabilities, leading to goals being evenly distributed across all percentiles.
By contrast, our trained models concentrate goals much more effectively toward the high-probability end of the distribution.
The **model trained on both distance and angle** (green curve) and only distance(blue curve) perform the best, showing a much steeper rise at the start , meaning it correctly ranks the majority of real goals in the top percentiles.
while the **angle-only model** (orange curve) is less steep and closer to the random baseline, suggesting weaker ability.


![ReliabilityDiagram](public/milestone2/image/baselineModels/ReliabilityDiagram.png)

The **reliability diagram** (or **calibration curve**) evaluates how well the model’s predicted probabilities reflect the actual likelihood of an event — in this case, whether a shot becomes a goal.

As shown in the figure above, our models demonstrate reasonable but imperfect calibration.
The **combined distance + angle model** (green curve) aligns most closely with the ideal diagonal, indicating that it produces the most reliable probability estimates among the compared models.
The **distance-only model** (blue curve) follows a similar trend, though slightly below the perfect calibration line, suggesting a mild tendency to underestimate goal probabilities.
In contrast, the **angle-only model** (orange curve) and the **random baseline** (red curve) deviate more substantially, revealing weaker calibration and poorer confidence alignment.

## <span style="color:#1E90FF;">Feature Engineering II Question</span>
code related to it: ift6758/data/feature_engineering_pt2.py ift6758/notebooks/feature_engineering_two.ipynb

In this section, we expanded our dataset by creating new engineered features. Here’s each feature’s name and simple explanation.


**previous_event_timeseconds** — Converts the timestamp of the previous event into total seconds.

**time_since_last_event** — Measures how much time has passed between the current event and the previous one.

**rebound** — Indicates whether the current shot happened right after a previous shot.

**distance_from_last_event** — The distance between the previous event’s coordinates and the current one.

**angle_shot_prev** — The angle from the net to the previous shot location.

**angle_shot** — The current shot’s angle relative to the net.

**speed** — The distance from the previous event, divided by the time since the previous event.

**angle_change** — The change in shot angle if the shot is a rebound.

**distance_shot** — The distance from the shooter’s position to the net.

**game_seconds_true** -  Calculates the true game event seconds given across the game of hockey.




## <span style="color:#1E90FF;">Advanced Models Question</span>
code related to it: ift6758/data/advanced_models.py ift6758/notebooks/advanced_models.ipynb

![XGBoostRoc](public/milestone2/image/advancedModels/XGBoostRoc.png)
![XGBoostGoalRate](public/milestone2/image/advancedModels/XGBoostGoalRate.png)
![XGBoostcumulative](public/milestone2/image/advancedModels/XGBoostcumulative.png)
![XGBoostAllReliability](public/milestone2/image/advancedModels/XGBoostReliability.png)
### XGBoost Baseline Model

To build a stronger baseline beyond logistic regression, we trained an **XGBoost classifier** using two engineered features: `distance_to_net` and `shot_angle`. We split the data into **80% for training** and **20% for validation**, using **2016–2017 to 2019–2020** seasons as training data and **2020–2021** as the testing season.

The **ROC curve** gives an **AUC of 0.711**, showing a small but clear improvement over the logistic regression model (AUC ≈ 0.701). This suggests that XGBoost can pick up more complex, non-linear relationships between distance, angle, and the chance of scoring.

The **Goal Rate vs. Predicted Probability Percentile** curve looks smooth and almost monotonic — higher predicted probabilities correspond to higher real goal rates. In other words, the model ranks shots in a logical way.

Looking at the **Cumulative Goals vs. Shots** plot, the curve rises quickly at the start, meaning the model does a good job identifying the most likely goals. Its performance here is similar to logistic regression.

Finally, the **Reliability Curve** shows that the predicted probabilities are mostly in line with the observed goal frequencies, though there’s some underestimation for high-probability bins (above 0.5).

**Overall**, XGBoost with just distance and angle already performs better than logistic regression — both in ranking and calibration. Next, we’ll see if adding more features can push the performance further.



### Data Preprocessing and Model Tuning

Before training the tuned XGBoost model, I performed several preprocessing steps to clean and transform the raw feature set.
The column `rebound` was converted to integer values, turning Boolean flags into numeric form. This ensures that the model interprets it as a binary variable (0 = no rebound, 1 = rebound) instead of a categorical string.

The dataset was then split by data type, and missing values were handled using different strategies.
For **numerical columns**, missing entries were filled with the **median** value to reduce the effect of outliers.
For **categorical columns**, missing entries were replaced with the placeholder 'Unknown', ensuring that no rows were dropped and all categories could still be encoded later.

Time-related columns such as `game_time` and `period_time` were converted from "MM:SS" string format into total seconds using helper functions:
X = adv._calculate_time_second(X)

X = adv._calculate_period_second(X)

After conversion, the original string columns were dropped.
We then standardized all numerical columns to keep continuous features on comparable scales and to avoid numerical instability during training.
Excluded columns such as shot_type, previous_event_name (categorical), and rebound (binary) were not scaled.

Next, **one-hot encoding** was applied to categorical variables like `shot_type` and `previous_event_name`.
Since these variables have no ordinal relationship between categories, one-hot encoding ensures that each type is treated as an independent feature.

After building a simple XGBoost model using only `distance_to_net` and `shot_angle` in the previous section, We extended the model by incorporating all engineered features — such as rebound indicators, speed, time since last event, and shot angle change.
To further improve the model’s predictive performance, we performed **hyperparameter tuning** using `RandomizedSearchCV` over a broad parameter space.
The parameters explored included:
- the number of estimators (`n_estimators`),
- maximum tree depth (`max_depth`),
- learning rate (`learning_rate`), and
- both L1/L2 regularization strengths (`reg_alpha`, `reg_lambda`).

Cross-validation was performed with **3 folds**, using **AUC** as the main evaluation metric to identify the best trade-off between model complexity and generalization ability.

---

### Why We Applied Calibration to XGBoost

Although XGBoost performs very well in terms of classification accuracy and ranking, its predicted probabilities are often **not well calibrated**.

To fix this, we applied a **calibration layer** on top of the trained XGBoost model using the `CalibratedClassifierCV` function from scikit-learn.
This extra layer adjusts the predicted probabilities to better reflect the observed goal frequencies.

After calibration, the **reliability curve** becomes more aligned with the diagonal line, meaning the model’s probabilities are more trustworthy.

After Data Preprocessing and Model Tuning, we got figures below.

![XGBoostRocAll](public/milestone2/image/advancedModels/XGBoostAllRoc.png)
![XGBoostGoalRateAll](public/milestone2/image/advancedModels/XGBoostAllGoalRate.png)
![XGBoostcumulativeAll](public/milestone2/image/advancedModels/XGBoostAllcumulative.png)
![XGBoostAllReliabilityAll](public/milestone2/image/advancedModels/XGBoostAllReliability.png)


After incorporating all engineered features — including rebound indicators, speed, time since last event, and shot angle change, among others — the **tuned XGBoost model** achieved a clear performance boost over the simpler version that used only distance and angle.
The **ROC curve** shows an increase in **AUC from 0.71 to 0.78**, reflecting stronger discriminative ability between goals and non-goals.
In the **Goal Rate vs Probability Percentile** plot, the curve becomes more sharply decreasing, meaning that high-probability predictions correspond more consistently to higher actual goal rates.
The **Cumulative Goals vs Shots** plot also steepens, showing that a smaller fraction of top-ranked shots now accounts for most goals — an indicator of improved model ranking.
Finally, the **Reliability Curve** aligns more closely with the perfect calibration line, confirming that the tuned XGBoost model provides better probability estimates and confidence calibration.

Overall, adding  features allows the model to capture more realistic hockey dynamics, improving both **accuracy** and **interpretability** compared to the baseline version.




### Feature Selection

Now let's  explore several **feature selection** techniques to check whether removing redundant or low-importance features could simplify the model and potentially improve its performance.

#### 1. Correlation-Based Filtering
We first examined **feature correlation** among numerical variables. Highly correlated features can introduce redundancy and may not provide new information to the model.
Using a correlation threshold of **0.8**, we identified the following pairs with near-perfect correlation: 'previous_event_timeseconds' with 'period_time_seconds' and 'previous_event_timeperiod_seconds' with 'period_time_seconds'.
After removing these redundant columns and retraining the model, the validation metrics remained almost unchanged:
**AUC = 0.7779**, **F1 = 0.303**, **Accuracy = 0.674**.
This indicated that these features contributed very little to the overall performance.


#### 2. Variance Threshold

Next, we applied a **VarianceThreshold** with a threshold of **0.01** to remove features with almost no variation across samples.
This method removed **8 low-variance features**, such as `shot_type_Unknown`, `previous_event_name_goal`, and `previous_event_name_penalty`.

After retraining the model, the performance slightly decreased:
**AUC = 0.7757**, **F1 = 0.3009**, **Accuracy = 0.672**.
Thus, low-variance features, although not individually important, still provided minor complementary information to the model.


#### 3. Mutual Information

To further understand feature importance, we used **Mutual Information (MI)**, which measures the dependency between each feature and the target variable.
The plot below shows that variables such as `shot_type_wrist`, `distance_shot`, `angle_shot`, and `num_friendly_skaters` had the highest mutual information scores, meaning they contributed the most predictive signal.

![mutualInformation](public/milestone2/image/advancedModels/mutualInformation.png)


However, removing features with low MI scores did not improve the model’s validation AUC.



#### 4. Recursive Feature Elimination (RFE)

Also we use another method **Recursive Feature Elimination (RFE)** with XGBoost as the base estimator, selecting the top **25 features**.
After retraining the model, the results were:

**AUC = 0.7777**, **F1 = 0.112**, **Accuracy = 0.910**.

Both the accuracy and  F1 score  decreased, showing that aggressive feature reduction caused the model to lose important interactions.


####  Final Observation

After testing multiple feature selection techniques — correlation filtering, variance thresholding, mutual information, and recursive feature elimination — we found that **keeping all engineered features produced the best overall performance**.

In conclusion, the **full feature set** achieves the most balanced and reliable results for this task.

#### 2.3 W&B Runs and Model References

Below are the **Weights & Biases (W&B)** run links that were used to generate the plots and metrics displayed above.
Each run corresponds to a specific methodology or model configuration evaluated in our research.

---

#### **XGB with distance and angle**
🔗 [View Run on W&B](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/artifacts/model/xgb_baseline/v1)

#### **XGB with all features**
🔗 [View Run on W&B](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/artifacts/model/xgb_all_features/v0)

#### **XGB with feature selection Variance Threshold**
🔗 [View Run on W&B](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/artifacts/model/xgb_selected_features_var/v0)

#### **XGB with feature selection Recursive Feature Elimination**
🔗 [View Run on W&B](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/artifacts/model/xgb_selected_features_rfe/v0)






## <span style="color:#1E90FF;">Give it your best shot Question</span>

code related to it: notebooks/give_it_your_best_shot.ipynb

### 1. Methodologies and Models Tested

These are the following plots for the methodologies tested.

Our first approach was to choose **six different classifiers** which we believed could have good performance on predicting if the shot was indeed a goal or not.
The models we chose were the following:
**RidgeClassifier**, **SGDClassifier**, **Random Forest**, **XGBoost**, **CatBoost**, and **MLP**.

Our first methodology was to simply do a **hyperparameter search**, so an exhaustive grid search on all possible combinations of hyperparameters which reduced the error on the validation set while increasing accuracy. All of the models had relative scores with one another, which will be shown in the following figures on which we performed the grid search.

[![Figure 1](public/milestone2/image/give_it_your_best_shot/figures/figure_1.png)](public/milestone2/image/give_it_your_best_shot/figures/figure_1.png)
[![Figure 2](public/milestone2/image/give_it_your_best_shot/figures/figure_2.png)](public/milestone2/image/give_it_your_best_shot/figures/figure_2.png)
[![Figure 3](public/milestone2/image/give_it_your_best_shot/figures/figure_3.png)](public/milestone2/image/give_it_your_best_shot/figures/figure_3.png)

The best model across all the testing was the **CatBoost classifier**.

> **Notice:**
> This pattern — where **CatBoost consistently outperformed** other models — was also observed in additional optimization tests that we conducted but did not include in this report.
> The full details of those experiments can be reviewed in the notebook  `give_it_your_best_shot.ipynb`

After obtaining our optimized hyperparameters, we also conducted other approaches which we found to be interesting to try out different models to increase accuracy while reducing error. To determine if one model was better than another, we chose to compare the **ROC score** instead of accuracy. This is an arbitrary metric we chose, but accuracy could have been chosen if desired.

Our second approach was to try **MRMR (Minimum Redundancy Maximum Relevance)** in which we observed the most relevant features in the dataframe while removing all the ones that weren't useful. This resulted in the following graph which shows the feature relevancy according to this methodology:

[![MRMR Feature Relevance](public/milestone2/image/give_it_your_best_shot/figures/mrmr_feature_relevance.png)](public/milestone2/image/give_it_your_best_shot/figures/mrmr_feature_relevance.png)

This kept 20 out of the 39 features and we then trained the model. This resulted in performance similar to the grid search method, and the models didn't yield any significant improvements.

Our third approach was to try a **Lasso regularization** method that we used on the training set, and we then selected the top 20 features, expanding from our last methodology. However, this also didn't yield any significant improvements or notable changes in model performance.

We also used Random Forest to find the best features in relation to the predicted value which is if the shot is a goal or not and this is the graph we obtain showing the relevant features:

[![Random Forest Feature Selected](public/milestone2/image/give_it_your_best_shot/figures/rf_feature_selection.png)](public/milestone2/image/give_it_your_best_shot/figures/rf_feature_selection.png)

We then trained a model with the top 20 features selected by this Random Forest method, but this led to no notable improvements in model performance.

We continued with other methods such as trying **different splits in the test and validation sets**, and **adding new features** such as `isRush`, which computes the time between the current and last event — if this difference in time is less than 4 seconds, we mark the event as a rush.
All these methodologies did not yield any significant differences in performance, which can be observed in the following plots and metrics for the given models.

Overall, our best model was indeed the **CatBoost classifier**, as when running it across all these different runs, CatBoost consistently came out on top in terms of performance — though it was always a close call between CatBoost and XGBoost.

---

### 2. Performance Plots and Metrics

These are the plots for the following curves for the respective methodologies which we found interesting to note:

####2.1 Plots
**ROC Curve (Distance + Angle)**
[![ROC Curve](public/milestone2/image/give_it_your_best_shot/roc_curve.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74/panel/jw48ews1m?nw=nwusergafranijaz)

**Goal Rate vs Probability Percentile**
[![Goal Rate](public/milestone2/image/give_it_your_best_shot/goal_rate.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74/panel/i367l86m7?nw=nwusergafranijaz)

**Cumulative Goals vs Shots**
[![Cumulative Goals](public/milestone2/image/give_it_your_best_shot/cumulative_goals.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74/panel/6np7s0575?nw=nwusergafranijaz)

**Reliability Curve – Validation**
[![Reliability](public/milestone2/image/give_it_your_best_shot/reliability_curve.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74/panel/b500ldcu6?nw=nwusergafranijaz)

#### 2.2 Metrics

Below are the comparative results for the four main methodologies tested in our research.
For each approach, we include three key evaluation plots — **ROC/AUC Curve**, **Brier Score**, and **Accuracy** — which help visualize model performance across different metrics.

---

#### **Grid Search**

**ROC / AUC Curve**
[![Grid Search ROC](public/milestone2/image/metrics/grid_search/gs_roc.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=gridsearch_best%2Froc_auc&panelSectionName=gridsearch_best)

**Brier Score**
[![Grid Search Brier](public/milestone2/image/metrics/grid_search/gs_brier.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=gridsearch_best%2Fbrier&panelSectionName=gridsearch_best)

**Accuracy**
[![Grid Search Accuracy](public/milestone2/image/metrics/grid_search/gs_accuracy.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=gridsearch_best%2Faccuracy&panelSectionName=gridsearch_best)

---

#### **MRMR (Minimum Redundancy Maximum Relevance)**

**ROC / AUC Curve**
[![MRMR ROC](public/milestone2/image/metrics/mrmr/mrmr_roc.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=mrmr_top_20%2Froc_auc&panelSectionName=mrmr_top_20)

**Brier Score**
[![MRMR Brier](public/milestone2/image/metrics/mrmr/mrmr_brier.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=mrmr_top_20%2Fbrier&panelSectionName=mrmr_top_20)

**Accuracy**
[![MRMR Accuracy](public/milestone2/image/metrics/mrmr/mrmr_accuracy.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=mrmr_top_20%2Faccuracy&panelSectionName=mrmr_top_20)

---

#### **Lasso Regularization**

**ROC / AUC Curve**
[![Lasso ROC](public/milestone2/image/metrics/lasso/lasso_roc.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=lasso_regularization%2Froc_auc&panelSectionName=lasso_regularization)

**Brier Score**
[![Lasso Brier](public/milestone2/image/metrics/lasso/lasso_brier.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=lasso_regularization%2Fbrier&panelSectionName=lasso_regularization)

**Accuracy**
[![Lasso Accuracy](public/milestone2/image/metrics/lasso/lasso_accuracy.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=lasso_regularization%2Faccuracy&panelSectionName=lasso_regularization)

---

#### **Split with 20 Features Removed**

**ROC / AUC Curve**
[![Split ROC](public/milestone2/image/metrics/split_20_features/split_roc.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=removed_20_features%2Froc_auc&panelSectionName=removed_20_features)

**Brier Score**
[![Split Brier](public/milestone2/image/metrics/split_20_features/split_brier.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=removed_20_features%2Fbrier&panelSectionName=removed_20_features)

**Accuracy**
[![Split Accuracy](public/milestone2/image/metrics/split_20_features/split_accuracy.png)](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/runs/26yweo74?nw=nwusergafranijaz&panelDisplayName=removed_20_features%2Faccuracy&panelSectionName=removed_20_features)

---

These are the scores for the respective methods we tried.
*Important to note that all these models are using **CatBoost**, and not any of the other five models mentioned beforehand.*

#### 2.3 W&B Runs and Model References

Below are the **Weights & Biases (W&B)** run links that were used to generate the plots and metrics displayed above.
Each run corresponds to a specific methodology or model configuration evaluated in our research.

---

#### **Grid Search**
🔗 [View Run on W&B](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/artifacts/model/gridsearch_best/v3)

#### **MRMR (Minimum Redundancy Maximum Relevance)**
🔗 [View Run on W&B](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/artifacts/model/mrmr_top_20/v3)

#### **Lasso Regularization**
🔗 [View Run on W&B](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/artifacts/model/lasso_regularization/v3)

#### **Split with 20 Features Removed**
🔗 [View Run on W&B](https://wandb.ai/IFT6758-2025-B1/IFT6758-2025-B01/artifacts/model/removed_20_features/v3)

---

*Each of the above runs contains detailed logs, metrics, and visualizations used in generating the figures for Sections 2.1 and 2.2.*


