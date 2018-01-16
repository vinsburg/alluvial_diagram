import numpy as np
from collections import Counter, defaultdict
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm
import itertools.product


def plot(input_data, *args, **kwargs):
    at = AlluvialTool(input_data, *args, **kwargs)
    ax = at.plot(**kwargs)
    ax.axis('off')
    return ax


class AlluvialTool:
    def __init__(
            self, input_data=(), x_range=(0, 1), res=20, h_gap_frac=0.03, v_gap_frac=0.03, **kwargs):
        _ = kwargs
        self.input = input_data
        self.x_range = x_range
        self.res = res  # defines the resolution of the splines for all veins
        self.trace_xy = self.make_vein_blueprint_xy_arrays()
        self.data_dic = self.read_input()
        self.item_widths_dic = self.get_item_widths_dic()
        self.a_members, self.b_members = self.get_item_groups(**kwargs)
        # self.a_members = sorted({a_item for a_item in self.data_dic}, reverse=True)
        # self.b_members = sorted(
        #     {b_item for b_item_counter in self.data_dic.values() for b_item in b_item_counter}, reverse=True)
        self.h_gap = x_range[1] * h_gap_frac
        self.v_gap_frac = v_gap_frac
        self.v_gap = sum(
            [width for b_item_counter in self.data_dic.values() for width in b_item_counter.values()]
        ) * v_gap_frac
        self.item_coord_dic = self.make_item_coordinate_dic()
        self.alluvial_fan = self.generate_alluvial_fan()

    def make_vein_blueprint_xy_arrays(self):
        y = np.array([0, 0.15, 0.5, 0.85, 1])
        x = np.linspace(self.x_range[0], self.x_range[-1], len(y))
        z = np.polyfit(x, y, 4)
        f = np.poly1d(z)

        blueprint_x_vals = np.linspace(x[0], x[-1], self.res)
        blueprint_y_vals = f(blueprint_x_vals)
        return blueprint_x_vals, blueprint_y_vals

    def get_vein_polygon_xy(self, y_range, width):
        x, y = self.trace_xy
        y0, yn = y_range
        scale = yn - y0
        ty = y * scale + y0
        x_new = np.concatenate([x, x[::-1], [x[0]]])
        y_new = np.concatenate([ty, ty[::-1] + width, [ty[0]]])
        return np.array([x_new, y_new]).transpose()
        # return x_new, y_new

    def read_input_from_table(self):
        data_table = np.array(self.input)
        data_dic = defaultdict(Counter)
        for line in data_table:
            data_dic[line[0]][line[1]] += 1
        return data_dic

    def read_input_from_dic(self):
        # data_dic = self.input
        # data_table = []
        # for x_item, y_item_counter in data_dic.items():
        #     for y_item, count in y_item_counter.items():
        #         data_table += [[x_item, y_item]] * count
        # data_table = np.array(sorted(data_table))
        # return data_table, data_dic
        return self.input

    def read_input(self):
        if type(self.input) == dict:
            return self.read_input_from_dic()
        else:
            return self.read_input_from_table()

    def get_item_widths_dic(self):
        iwd = Counter()
        for a_item, b_item_counter in self.data_dic.items():
            for b_item, width in b_item_counter.items():
                iwd[a_item] += width
                iwd[b_item] += width
        return iwd

    def get_item_groups(self, a_sort=None, b_sort=None, **kwargs):
        _ = kwargs
        a_members = sorted(
            {a_item for a_item in self.data_dic}, key=lambda x: self.item_widths_dic[x]
        ) if not a_sort else a_sort
        b_members = sorted(
            {b_item for b_item_counter in self.data_dic.values() for b_item in b_item_counter},
            key=lambda x: self.item_widths_dic[x]
        ) if not b_sort else b_sort
        return a_members, b_members

    def make_item_coordinate_dic(self, ):
        item_coord_dic = defaultdict(ItemCoordRecord)
        groups = self.a_members, self.b_members
        group_widths = [self.get_group_width(group) for group in groups]
        for ind, group in enumerate(groups):
            last_pos = (max(group_widths) - group_widths[ind]) / 2
            for item in group:
                width = self.item_widths_dic[item]
                xy = (self.x_range[ind], last_pos)
                item_coord_dic[item].set_start_state(width, xy, side=ind)
                last_pos += width + self.v_gap
        return item_coord_dic

    def get_group_width(self, group):
        return sum([self.item_widths_dic[item] for item in group]) + (len(group) - 1) * self.v_gap

    def generate_alluvial_vein(self, a_item, b_item):
        width = self.data_dic[a_item][b_item]
        a_item_coord = self.item_coord_dic[a_item].read_state_and_advance_y(width)
        b_item_coord = self.item_coord_dic[b_item].read_state_and_advance_y(width)
        y_range = (a_item_coord[1], b_item_coord[1],)
        return self.get_vein_polygon_xy(y_range, width)

    def get_label_rectangles_xy(self, a_item, b_item):
        width = self.data_dic[a_item][b_item]
        return (
            self.generate_item_sub_rectangle(a_item, width),
            self.generate_item_sub_rectangle(b_item, width),
        )

    def generate_item_sub_rectangle(self, item, width):
        dic_entry = self.item_coord_dic[item]
        item_coord = dic_entry.read_state()
        sign = dic_entry.get_side_sign()
        return self.get_rectangle_xy(item_coord, width, sign)

    def get_rectangle_xy(self, item_coord, width, sign):
        x, y = item_coord
        rect = [[
                    x + sign * 0.5 * (0.5 + xa) * self.h_gap,
                    y + ya * width,
                ] for xa, ya in itertools.product((0, 1), repeat=2)]
        rect = [rect[0]] + [rect[1]] + [rect[3]] + [rect[2]] + [rect[0]]
        return np.array(rect)

    def generate_alluvial_fan(self, ):
        alluvial_fan = []
        for a_item in self.a_members:
            b_items4a_item = self.data_dic[a_item].keys()
            for b_item in self.b_members:
                if b_item in b_items4a_item:
                    l_a_rect, l_b_rect = self.get_label_rectangles_xy(a_item, b_item)
                    alluvial_fan += [
                        [self.generate_alluvial_vein(a_item, b_item), l_a_rect, l_b_rect, a_item, b_item, ]]
        return np.array(alluvial_fan)

    def plot(self, cmap=matplotlib.cm.get_cmap('jet'), figsize=(10, 15), alpha=0.4, **kwargs):
        colors = self.get_random_colors(**kwargs)
        fig, ax = plt.subplots(figsize=figsize)
        for num in (0, 1, 2):
            patches = [Polygon(item) for item in self.alluvial_fan[:, num]]
            pc = PatchCollection(patches, cmap=cmap, alpha=alpha)
            pc.set_array(np.array(colors))
            ax.add_collection(pc)
        self.auto_label_veins(**kwargs)
        ax.autoscale()
        return ax

    def get_random_colors(self, color_side=0, rand_seed=0, **kwargs):
        _ = kwargs
        color_items = self.b_members if color_side else self.a_members
        np.random.seed(rand_seed)
        color_array = np.random.rand(len(color_items))
        ind_dic = {item: ind for ind, item in enumerate(color_items)}
        colors = []
        for _, _, _, a_item, b_item, in self.alluvial_fan:
            item = b_item if color_side else a_item
            colors += [color_array[ind_dic[item]]]
        return np.array(colors)

    def auto_label_veins(self, **kwargs):
        # shift = max([len(item) for item in self.item_coord_dic.keys()]) / 50
        text_kwargs = self.get_text_kwargs(kwargs)
        for item, vein in self.item_coord_dic.items():
            y_width = vein.get_width()
            sign = vein.get_side_sign()
            ha = 'right' if sign == -1 else 'left'
            plt.text(
                vein.get_x() + 1.5 * sign * self.h_gap,
                vein.get_y() + y_width / 2,
                self.item_text(item, **kwargs),
                ha=ha, va='center', **text_kwargs)

    def item_text(self, item, show_width=False, **kwargs):
        _ = kwargs
        if show_width:
            ans = '{} - {}'.format(item, self.item_coord_dic[item].get_width())
        else:
            ans = '{}'.format(item)
        return ans

    @staticmethod
    def get_text_kwargs(kwargs):
        kw_list = ['agg_filter',
                   'alpha',
                   'animated',
                   'backgroundcolor',
                   'bbox',
                   'clip_box',
                   'clip_on',
                   'clip_path',
                   'color',
                   'contains',
                   'family',
                   'fontfamily',
                   'fontname',
                   'name',
                   'figure',
                   'fontproperties',
                   'font_properties',
                   'gid',
                   'horizontalalignment',
                   'ha',
                   'label',
                   'linespacing',
                   'multialignment',
                   'ma',
                   'path_effects',
                   'picker',
                   'position',
                   'rasterized',
                   'rotation',
                   'rotation_mode',
                   'size',
                   'fontsize',
                   'sketch_params',
                   'snap',
                   'stretch',
                   'fontstretch',
                   'style',
                   'fontstyle',
                   'text',
                   'transform',
                   'url',
                   'usetex',
                   'variant',
                   'fontvariant',
                   'verticalalignment',
                   'va',
                   'visible',
                   'weight',
                   'fontweight',
                   'wrap',
                   'x',
                   'y',
                   'zorder',
                   ]
        return {key: value for key, value in kwargs.items() if key in kw_list}


class ItemCoordRecord:
    def __init__(self, ):
        self.width = 0
        self.xy = ()
        self.curr_xy = self.xy[:]
        self.side = -1

    def set_start_state(self, width, xy, side):
        self.width = width
        self.xy = xy
        self.curr_xy = list(self.xy[:])
        self.side = side

    def read_state_and_advance_y(self, width):
        out = self.curr_xy[:]
        self.curr_xy[1] += width
        return out

    def read_state_and_advance_x(self, width):
        out = self.curr_xy[:]
        self.curr_xy[0] += width
        return out

    def read_state(self):
        return self.curr_xy[:]

    def get_xy(self, ):
        return self.xy

    def get_x(self, ):
        return self.xy[0]

    def get_y(self, ):
        return self.xy[1]

    def get_width(self, ):
        return self.width

    def get_side_sign(self, ):
        return 1 if self.side else -1
