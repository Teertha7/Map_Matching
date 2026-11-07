# Test whether candidate roads are labelled correctly

target = 123
ranked = [
    (123, 5.2),
    (456, 8.1),
    (789, 12.4),
    (101, 18.6),
]

print("Target road ID:", target)
print("Candidate road IDs:", [cid for cid, dist in ranked])

labels = [int(cid == target) for cid, dist in ranked]

print("Generated labels:", labels)

assert labels.count(1) == 1, "ERROR: Expected exactly one positive label!"
assert labels.count(0) == len(ranked) - 1, "ERROR: Negative labels are missing!"

print("SUCCESS: Candidate labels are correct.")