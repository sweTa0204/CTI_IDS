# 🎯 XAI PRESENTATION - ONLY 3 IMAGES

## Location: `FINAL/XAI_PPT_IMAGES/`

---

# IMAGE 1: `global_importance_bar.png`
## SHAP Feature Importance (Global View)

### What It Shows:
A bar chart ranking which features are MOST important for detecting DoS attacks.

### How to Explain:
> "This chart shows SHAP's analysis of what features the model looks at most when deciding if something is a DoS attack.
>
> The top features are:
> - **dmean** (destination mean packet size) - when server is flooded, it can barely respond
> - **sload** (source load) - how fast data is being sent
> - **sbytes** (source bytes) - total amount of data sent
>
> These make sense because DoS attacks work by sending LOTS of data VERY FAST to overwhelm servers."

### One-Line Summary:
> "SHAP tells us WHAT features matter most - and they all relate to traffic volume and speed, which is exactly how DoS attacks work."

---

# IMAGE 2: `xgb_waterfall_sample.png`
## SHAP Waterfall (Single Prediction Explanation)

### What It Shows:
How ONE specific network connection was classified, showing each feature's contribution.

### How to Explain:
> "This is explaining ONE prediction. 
>
> - It starts from a base value (middle - 50% probability)
> - Each feature either PUSHES toward DoS (red/right) or Normal (blue/left)
> - The final prediction is the sum of all these pushes
>
> For this sample:
> - High sload pushed +15% toward DoS
> - High sbytes pushed +12% toward DoS
> - Normal proto pushed -3% toward Normal
> - Final result: DoS attack detected
>
> This is how we explain WHY a specific alert was raised."

### One-Line Summary:
> "SHAP waterfall shows HOW each feature contributed to THIS specific prediction - security analysts can see exactly WHY an alert was triggered."

---

# IMAGE 3: `xgb_lime_sample.png`
## LIME Explanation (Simple Rules)

### What It Shows:
Simple IF-THEN rules that explain the prediction in human-readable format.

### How to Explain:
> "LIME creates simple rules that anyone can understand:
>
> - IF sbytes > 500,000 → likely DoS
> - IF sload > 2,000,000 → likely DoS
> - IF rate > 5,000 → likely DoS
>
> For this connection, the green bars show features pushing toward the predicted class.
>
> The difference from SHAP:
> - SHAP: Mathematical - 'sbytes contributed +0.15'
> - LIME: Rule-based - 'IF sbytes > threshold THEN DoS'"

### One-Line Summary:
> "LIME gives human-readable rules - even a non-technical security analyst can understand 'high bytes + high speed = attack'."

---

# 🗣️ WHAT TO SAY IN REVIEW (3 MINUTES)

## Slide 1: Feature Importance
> "First, SHAP tells us WHAT the model looks at. The top features are all related to data volume and speed - exactly what DoS attacks exploit."

## Slide 2: Waterfall
> "Second, for any specific prediction, SHAP shows HOW it was made. Each feature adds or subtracts from the probability. This lets us verify the model's reasoning."

## Slide 3: LIME
> "Finally, LIME gives simple rules. 'IF bytes are high AND speed is high THEN DoS.' This is what we show to security analysts - they can immediately understand and trust the alert."

## Conclusion:
> "Together, SHAP and LIME make our DoS detection EXPLAINABLE. Not just 'this is an attack' but 'this is an attack BECAUSE of these specific reasons.' This is critical for real-world deployment where humans need to trust and act on alerts."

---

# ✅ CHECKLIST

Your 3 images are in: `FINAL/XAI_PPT_IMAGES/`

| # | Image | Purpose |
|---|-------|---------|
| 1 | `global_importance_bar.png` | WHAT features matter |
| 2 | `xgb_waterfall_sample.png` | HOW one prediction works |
| 3 | `xgb_lime_sample.png` | SIMPLE rules for humans |

That's it! 3 images, 3 minutes, complete XAI explanation.
