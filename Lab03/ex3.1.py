from numpy import random
def choose_player():
    coin = random.randint(1, 3)
    if coin == 1:
        return False
    else:
        return True

def dice_roll():
    return random.randint(1, 6)

def flip_coin():
    player = choose_player()
    dice = dice_roll()
    m = 0
    prob = 0.5
    if player == False:
        prob = 0.57
    for i in range(1, dice*2):
        flip = random.randint(1, 100)
        if flip <= prob*100 :
            m += 1
    if dice >= m:
        return player
    else:
        return not player

def estimate():
    count = 0
    for each in range(10000):
        player = flip_coin()
        if player == 1:
            count += 1

    if count > 5000:
        print("A castigat player 1, cu probabilitate: ", count/10000)
    else: print("A castigat player 0, cu probabilitate: ", (1 - count/10000))


estimate()
