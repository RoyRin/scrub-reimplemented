# Reimplementing SCRUB (Machine Unlearning) 

https://arxiv.org/abs/2302.09880


High-level intuition of the algorithm

SCRUB:
  1. create teacher (original model), and student (copy of original model)
  2. iteratively:
    * on forget set - maximize KL divergence between teacher probabilities and student probabilities  (for each point)
    * on retain set - minimize KL divergence between teacher and student (for each point)
    * on retain set - minimize test loss
  3. return student
