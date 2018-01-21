# alluvial_diagram
A python script for generating "alluvial" styled bipartite diagrams, using matplotlib and numpy

## Getting Started

Copy alluvial.py to your working directory, and follow the syntax in the examples below.
See "Advanced use" for a mild parameter documentation.

### Prerequisites

matplotlib and numpy

### Installing

Copy alluvial.py to your working directory.

#### Example 1:
<pre><code>
import alluvial
import matplotlib.pyplot as plt
import numpy as np

input_data = {'a': {'aa': 0.3, 'cc': 0.7,},
              'b': {'aa': 2, 'bb': 0.5,},
              'c': {'aa': 0.5, 'bb': 0.5, 'cc': 1.5,}}

ax = alluvial.plot(input_data)
fig = ax.get_figure()
fig.set_size_inches(5,5)
plt.show()
</code></pre>
![](/image_examples/Example1.png)

#### Example 2:
<pre><code>
import alluvial
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

# Generating the input_data:
seed=7
np.random.seed(seed)
def rand_letter(num): return chr(ord('A')+int(num*np.random.rand()))

input_data = [[rand_letter(15), rand_letter(5)*2] for _ in range(50)]

# Plotting:
cmap = matplotlib.cm.get_cmap('jet')
ax = alluvial.plot(
    input_data,  alpha=0.4, color_side=1, rand_seed=seed, figsize=(7,5),
    disp_width=True, wdisp_sep=' '*2, cmap=cmap, fontname='Monospace',
    labels=('Capitals', 'Double Capitals'), label_shift=2)
ax.set_title('Utility display', fontsize=14, fontname='Monospace')
plt.show()
</code></pre>
![](/image_examples/Example2.png)

### Advanced use
* Additional input format - a list of tuples of structure:
  * input_data = [('a_item0', 'b_item0'), ('a_item0', 'b_item1') , ('a_item1', 'b_item0')]
* Parameters and default values:
  * alpha=0.5 - defines facecolor alpha for all veins
  * color_side=0 - vein colors determined by left side items (0) or right side items (1)
  * x_range=(0, 1) - changes the horizontal plot coordinates
  * res=20 - determines the number of points constituting the alluvial vein spline
  * h_gap_frac=0.03 - changes the horizontal gap between the labels, vein base rectangles, and veins
  * v_gap_frac=0.03 - changes the vertical gap between veins
  * colors=None - an optional list of matplotlib spec colors, len(colors) must be equal to the number of items on color_side
  * cmap=None - a matplotlib.cm color map instance, for choosing random colors. if None, 'hsv' is used
  * rand_seed=1 - a seed for the random color generator, if None colors are chosen at random
  * a_sort, b_sort - lists defining plot order of items (both are None by default). if None, items are sorted by width
  * disp_width=False - if True, displays vein widths beside item labels
  * wdisp_sep=7*' ' - seperates width and item text if width is displayed
  * width_in=True - displays width between the item text and the graph, reversed order if False
  * labels=None - a tuple of form ('a_label', 'b_label'), if None side labels are not plotted
  * figsize=(10, 15), The figure size.
  * fontname='Arial' The font of all figure text


## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Inspired by the alluvial diagram example given in the [rawgraphs website](http://rawgraphs.io/gallery_project/visualizations-for-issue-mapping-book/).


