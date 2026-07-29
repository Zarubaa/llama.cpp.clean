# INVALID NORMAL-REQUEST LABEL

This five-case sweep had stable GPU compute conditions and is useful only for
synthetic-path debugging. It was run before the prompt-tokenization fix: the
10200-token prompt overflowed a fixed 1152-token buffer, after which the tool
silently generated sequential synthetic token IDs. It is not a normal-request
result and must not be used in the final comparison.
