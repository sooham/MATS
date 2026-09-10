# Qwen returns unnormalized pre-softmax scores, so individual raw logits do not
# have a logarithm base.  PyTorch's softmax uses exp(), however, which makes
# log-probabilities and logit differences natural-log quantities (nats).
SOFTMAX_LOG_BASE = "e"
SOFTMAX_LOG_UNIT = "nats"

YES = "YES"
NO = "NO"