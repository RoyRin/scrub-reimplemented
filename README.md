# Reimplementing SCRUB (Machine Unlearning) 

https://arxiv.org/abs/2302.09880

This code was created and made public because I was not able to get the the original codebase by the authors to work (https://github.com/meghdadk/SCRUB/tree/main). 
I've taken `thirdparty/repdistiller` mostly [from their code](https://github.com/meghdadk/SCRUB/tree/main/thirdparty/repdistiller). 

## How to run
run `scrub_eval.py` or `scrub_eval.ipynb`

## High-level intuition of the algorithm

SCRUB:
  1. create teacher (original model), and student (copy of original model)
  2. iteratively:
      
      1. on forget set - maximize KL divergence between teacher probabilities and student probabilities  (for each point)
      2. on retain set - minimize KL divergence between teacher and student (for each point)
      3. on retain set - minimize test loss
  
  6. return student
