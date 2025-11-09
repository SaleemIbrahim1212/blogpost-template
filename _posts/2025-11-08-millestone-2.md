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




## <span style="color:#1E90FF;">Advanced Models Question</span>
code related to it: ift6758/data/advanced_models.py ift6758/notebooks/advanced_models.ipynb


