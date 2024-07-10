# Model Hyperparamets

## Fine tuning
- Regarding the head layer (`fc`): currently we have a single 512 -> 1 linear layer.  
  We could also have more than one layer in the head, what are the pros and cons?
- Currently all other layers are frozen. Should we perhaps unfreeze some of them? what are the common strategies and their tradeoffs?

## Optimizer
Currently we have Adam as the optimizer with `learning_rate=3e-4`.
Does it have other parameters? is there another optimizer that might be useful (e.g. AdamW)?

## Scheduler
Currently there's no scheduler. What are the options and their tradeoffs?

## Loss function
Our loss function, focal loss, has two parameters - `alpha` and `gamma`.
Currently we have `alpha = 1/6` approximately, `gamma = 2`. Are these good picks?

# Score function
Our current score function is MCC (at this specific moment, training a model on accuracy instead as a sanity check).

# Others
- What about number of epochs?
- We should also consult the parameters that the original ResNet (2+1)D net was trained on.  
  How do they compare to our preprocessed data? do we need to change anything?