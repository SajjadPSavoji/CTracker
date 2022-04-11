import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# import parse
def parse_it(line):
   return [float(part.split(":")[1]) for part in line[:-1].split("|")]

def averaged(loss_arr):
    return [ np.average(loss_arr[:i+1]) for  i in range(len(loss_arr))]
log_dir = "./log.txt"
lines = None
with open(log_dir, "r") as f:
    lines = f.readlines()
# x = parse.parse(lines[0], 'Epoch: {} | Iter: {} | Cls loss: {:1.5f} | Reid loss: {:1.5f} | Reg loss: {:1.5f} | Running loss: {:1.5f}')
losses = np.array([parse_it(line) for line in lines])
# idx = 2 -> cls losses
# idx = 3 -> reId loss 
# idx = 4 -> Reg loss
# idx = 5 -> running loss
epochs = len(losses)/1993
x = np.linspace(0, epochs, len(losses))


plt.plot(x, averaged(losses[:, 2]), label = "cls")
plt.plot(x, averaged(losses[:, 3]), label = "reID")
plt.plot(x, averaged(losses[:, 4]), label = "Reg")
plt.plot(x, averaged(losses[:, 5]), label = "Avg")
plt.grid()
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("./Training_Curve.png")