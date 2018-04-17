import numpy as np
from matplotlib import pyplot as plt
from subprocess import Popen, PIPE


def run(path) -> str:
    """
метод для того, чтобы запускать ехешник из питона
    """
    p = Popen(path, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = p.communicate()
    if p.returncode == 0:
        print("Executed successfully")
        return out.decode("cp866")
    else:
        print("Error:")
        return err.decode("cp866")


def process_output(out: str) -> (list, list, list):
    out = out.splitlines()
    print("Completed in " + out[0] + " sec")
    x_s = np.fromstring(out[1], dtype=int, sep=' ')
    y_s = np.fromstring(out[2], dtype=int, sep=' ')
    rgb_str_list = out[3].split("|")
    rgb_str_list.pop()
    rgb_s = []
    for item in rgb_str_list:
        rgb_s.append([float(s) for s in item.split(' ')])
    return x_s, y_s, rgb_s


x, y, rgb = process_output(run("\cuda\\x64\Debug\cuda.exe"))
# x, y, rgb = process_output(run("\openmp\\x64\Debug\SOM.exe"))

fig, ax = plt.subplots()
ax.scatter(x, y, s=10, facecolors=np.array(rgb))
plt.show()

