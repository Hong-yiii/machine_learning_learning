import torch
from math import exp, log

# softmax function for multiclass classification --> gives a list of "probabilites" that adds up to 1, kinda based of confidence ig

def softmax(logits):
  counts = [logit.exp() for logit in logits]
  denominator = sum(counts)
  out = [c / denominator for c in counts]
  return out

# negative log likelihood loss function (copy pasted from tutorial)
# logits = [Value(0.0), Value(3.0), Value(-2.0), Value(1.0)]
# probs = softmax(logits)
# loss = -probs[3].log()

logits = [torch.tensor([0.0]).double(), torch.tensor([3.0]).double(), torch.tensor([-2.0]).double(), torch.tensor([1.0]).double(),]

# turn on gradient for everything (pytorch removes leaf nodes as .grad is usually not needed -> removed for efficiency)
for logit in logits:
  logit.requires_grad = True

probs = softmax(logits)
loss = -probs[3].log()

# print loss
print(loss.data.item())

# backpropagate and evaluate answer
loss.backward()

ans = [0.041772570515350445, 0.8390245074625319, 0.005653302662216329, -0.8864503806400986]


for dim in range(4):
    ok = 'OK' if abs(logits[dim].grad.item() - ans[dim]) < 1e-5 else 'WRONG!'
    print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {logits[dim].grad.item()}")
