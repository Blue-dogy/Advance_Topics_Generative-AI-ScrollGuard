# EXAONE 3.5 7.8B results before the field-description fix

These runs used the prompt where `usage_quadratic` was described as "Usage concentration (0-1)".
EXAONE read it as an attention score ("high concentration level ... indicates intentional study") in 72% of its track-A reasons, although the column measures how much of the hour's use falls in one long session.
The description was clarified afterwards, so these files are kept only as a record and are not used for the final comparison.

| Track | Runs | Accuracy | Cost score | Notes |
|---|---|---|---|---|
| A | 3 | 0.63 | 0.72 | counterfactual pairs 1.00, never chose `ask_context` |
| B | 1 | 0.36 | 0.51 | mostly `no_intervention`; read uniform swipes as engaged viewing; checked user context in 40% of cases |
