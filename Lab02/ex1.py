from numpy import random
def add_ball():
    """
    1 - red
    2 - blue
    3 - black
    """
    urn = [1, 1, 1, 2, 2, 2, 2, 3, 3]
    dice = random.randint(1, 6)
    if dice == 6:
        urn.append(1)
    if dice == 1 or dice == 4:
        urn.append(2)
    if dice == 5 or dice == 2 or dice == 3:
        urn.append(3)

    return urn

def extract_ball():
    urn = add_ball()
    ballin = random.randint(1, 10)
    ball = urn[ballin-1]
    return ball

def estimate():
    count = 0
    for each in range(1000):
        ball = extract_ball()
        if ball == 1:
            count += 1

    return count/1000


estimate = estimate()
print("Probability: ~0.32")
print("Estimate:", estimate)