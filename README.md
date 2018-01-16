# alluvial_diagram
A python script for generating "alluvial" styled bipartite diagrams, using matplotlib and numpy

## Example1:
<pre><code>
import alluvial
import matplotlib.pyplot as plt
import numpy as np

input_data = {'a': {'aa': 0.3, 'cc': 0.7,},
              'b': {'aa': 2, 'bb': 0.5,},
              'c': {'aa': 0.5, 'cc': 1.5,}}
ax = alluvial.plot(input_data)
fig = ax.get_figure()
fig.set_size_inches(5,5)
plt.show()
</code></pre>
![Alt text](/image_examples/Example1.png)

## Example2:
<pre><code>
import alluvial
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

def rand(mm):
    return int((mm**2)*np.random.rand()//mm)
mm = 5
input_data = {chr(rand(mm*4)+65): {chr(rand(mm)+97): 4*rand(mm) for _ in range(mm)} for _ in range(mm*4)}

cmap = matplotlib.cm.get_cmap('gist_rainbow')
ax = lib.figs.alluvial.plot(input_data,  alpha=0.6, color_side=1, rand_seed=1, show_width=True, figsize=(10,15), cmap=cmap)
fig = ax.get_figure()
plt.show()
</code></pre>
![Alt text](/image_examples/Example2.png)
