import sys

def printProgressBar(value: float, label: str) -> None:
    animation = '|/-\\'
    n_bar = 40 # tamanho da barra
    max = 100
    j = value / max
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1 - j))
    sys.stdout.write('\r')
    sys.stdout.write(f"{label.ljust(10)}  {animation[int(100 * j) % len(animation) if int(100 * j) != 100 else 0]}  [{bar:{n_bar}s}]  {int(100 * j)}% ")
    sys.stdout.flush()

def printAnimatedBar(value: float, label: str) -> None:
    animation1 = '|/-\\'
    animation2 = [
        "[-     ]",
        "[ -    ]",
        "[  -   ]",
        "[   -  ]",
        "[    - ]",
        "[     -]",
        "[    - ]",
        "[   -  ]",
        "[  -   ]",
        "[ -    ]"
    ]
    animation3 = [
        "[        ]",
        "[-       ]",
        "[--      ]",
        "[---     ]",
        "[----    ]",
        "[-----   ]",
        "[------  ]",
        "[------- ]",
        "[--------]",
        "[ -------]",
        "[  ------]",
        "[   -----]",
        "[    ----]",
        "[     ---]",
        "[      --]",
        "[       -]"
    ]
    animation4 = [
        "    ",
        "░   ",
        "▒   ",
        "▓   ",
        "█   ",
        "█░  ",
        "█▒  ",
        "█▓  ",
        "██  ",
        "██░ ",
        "██▒ ",
        "██▓ ",
        "███ ",
        "███░",
        "███▒",
        "███▓",
        "████",
        "▓███",
        "▒███",
        "░███",
        " ███",
        " ▓██",
        " ▒██",
        " ░██",
        "  ██",
        "  ▓█",
        "  ▒█",
        "  ░█",
        "   █",
        "   ▓",
        "   ▒",
        "   ░"
    ]
    animation = animation3
    max = 100
    j = value / max
    sys.stdout.write('\r')
    sys.stdout.write(f"{label.ljust(10)}  {animation[int(100 * j) % len(animation) if int(100 * j) != 100 else 8]}  {int(100 * j)}% ")
    sys.stdout.flush()
