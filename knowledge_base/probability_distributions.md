# Probability: Distributions

## Binomial
- \( n \) trials, success probability \( p \). \( P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} \)
- Mean \( \mu = np \), variance \( \sigma^2 = np(1-p) \).

## Common pitfalls
- "At least one" = 1 - P(none).
- "Exactly k" use binomial formula; "at most k" sum over 0 to k.
- Check whether order matters (permutations) or not (combinations).
