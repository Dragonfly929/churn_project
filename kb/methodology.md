# Methodology of the churn early-warning system

## Risk score
The score is the probability of leaving in the next 90 days, from a gradient boosting model trained on the Stage 1 features of the data warehouse. The alert threshold comes from the cost matrix: a lost client costs about 1000 EUR, a useless offer about 60 EUR.

## Silence detection
Silence sends no events. Therefore every active client is rescored when a bank day closes, even with no new event. Days without activity increase and the risk rises without any message.

## Smoothing
Daily scores are smoothed with an exponentially weighted moving average, so a single noisy day does not trigger an alert.

## Limits
The bank and the clients' reactions are simulated; response rates of offers are unknown to the policy and learned with Thompson Sampling.
