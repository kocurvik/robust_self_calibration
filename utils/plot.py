import numpy as np
from matplotlib import pyplot as plt
from seaborn.categorical import _BoxPlotter
from seaborn.utils import remove_na


class _CustomBoxPlotter(_BoxPlotter):
    def draw_boxplot(self, ax, kws):
        """Use matplotlib to draw a boxplot on an Axes."""
        vert = self.orient == "v"

        props = {}
        for obj in ["box", "whisker", "cap", "median", "flier"]:
            props[obj] = kws.pop(obj + "props", {})

        max_lens = []
        # calculate max num of levels
        for i, group_data in enumerate(self.plot_data):
            max_len = 0
            for j, hue_level in enumerate(self.hue_names):
                if group_data.size == 0:
                    continue

                hue_mask = self.plot_hues[i] == hue_level
                box_data = np.asarray(remove_na(group_data[hue_mask]))
                if box_data.size == 0:
                    continue

                max_len += 1

            max_lens.append(max_len)
        max_levels = np.max(max_lens)


        for i, group_data in enumerate(self.plot_data):
            # Draw nested groups of boxes
            # n_levels = len(self.hue_names)
            # each_width = self.width / (n_levels - 1)
            # offsets = np.linspace(0, self.width - each_width, n_levels - 1)
            # offsets -= offsets.mean()
            # offsets = np.append(offsets, 0.0)
            # widths = [self.width / len(self.hue_names) * .98 for _ in range(n_levels)]
            # # widths.append(self.width * 0.98)

            box_datas = []
            colors = []

            for j, hue_level in enumerate(self.hue_names):

                # Add a legend for this hue level
                if not i:
                    self.add_legend_data(ax, self.colors[j], hue_level)

                # Handle case where there is data at this level
                if group_data.size == 0:
                    continue

                hue_mask = self.plot_hues[i] == hue_level
                box_data = np.asarray(remove_na(group_data[hue_mask]))

                # Handle case where there is no non-null data
                if box_data.size == 0:
                    continue

                box_datas.append(box_data)
                colors.append(j)

            n_levels = len(box_datas)
            each_width = self.width / (n_levels)
            offsets = np.linspace(0, self.width - each_width, n_levels)
            offsets -= offsets.mean()
            offsets = np.append(offsets, 0.0)
            widths = [self.width / max_levels * .9 * self.width for _ in range(n_levels)]
            # widths = [self.width / n_levels * .9 for _ in range(n_levels)]
            # widths.append(self.width * 0.98)

            for j in range(len(box_datas)):
                center = i + offsets[j]
                artist_dict = ax.boxplot(box_datas[j],
                                         vert=vert,
                                         patch_artist=True,
                                         positions=[center],
                                         widths=widths[j],
                                         **kws)
                self.restyle_boxplot(artist_dict, self.colors[colors[j]], props)
                # Add legend data, but just for one set of boxes
    def restyle_boxplot(self, artist_dict, color, props):
        """Take a drawn matplotlib boxplot and make it look nice."""
        for box in artist_dict["boxes"]:
            box.update(dict(facecolor=tuple(0.7*x + 0.3 for x in color),
                            zorder=.9,
                            edgecolor=color,
                            linewidth=self.linewidth))
            box.update(props["box"])
        for whisk in artist_dict["whiskers"]:
            whisk.update(dict(color=color,
                              linewidth=self.linewidth,
                              linestyle="-"))
            whisk.update(props["whisker"])
        for cap in artist_dict["caps"]:
            cap.update(dict(color=color,
                            linewidth=self.linewidth))
            cap.update(props["cap"])
        for med in artist_dict["medians"]:
            med.update(dict(color=color,
                            zorder=1.0,
                            linewidth=self.linewidth))
            med.update(props["median"])
        for fly in artist_dict["fliers"]:
            fly.update(dict(markerfacecolor=color,
                            marker="d",
                            markeredgecolor=color,
                            markersize=self.fliersize))
            fly.update(props["flier"])


def custom_dodge_boxplot(
    *,
    x=None, y=None,
    hue=None, data=None,
    order=None, hue_order=None,
    orient=None, color=None, palette=None, saturation=.75,
    width=.8, dodge=True, fliersize=5, linewidth=None,
    whis=1.5, ax=None,
    **kwargs
):

    plotter = _CustomBoxPlotter(x, y, hue, data, order, hue_order,
                                    orient, color, palette, saturation,
                                    width, dodge, fliersize, linewidth)

    if ax is None:
        ax = plt.gca()
    kwargs.update(dict(whis=whis))

    plotter.plot(ax, kwargs)
    return ax


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_scene(points, R, t, f1, f2, width=640, height=480, color_1='black', color_2='red', name=""):
    c_x_1 = np.array([0.5 * width, 0.5 * width, -0.5 * width, -0.5 * width, 0])
    c_y_1 = np.array([0.5 * height, -0.5 * height, -0.5 * height, 0.5 * height, 0])
    c_z_1 = np.array([f1, f1, f1, f1, 0])
    c_z_2 = np.array([f2, f2, f2, f2, 0])

    camera2_X = np.row_stack([c_x_1, c_y_1, c_z_2, np.ones_like(c_x_1)])
    c_x_2, c_y_2, c_z_2 = np.column_stack([R.T, -R.T @ t]) @ camera2_X

    # fig = plt.figure()

    ax = plt.axes(projection="3d")
    ax.set_box_aspect([1.0, 1., 1.0])

    ax.plot3D(c_x_1, c_y_1, c_z_1, color_1)
    ax.plot3D(c_x_2, c_y_2, c_z_2, color_2)

    ax.plot3D([c_x_1[0], c_x_1[3]], [c_y_1[0], c_y_1[3]], [c_z_1[0], c_z_1[3]], color_1)
    ax.plot3D([c_x_2[0], c_x_2[3]], [c_y_2[0], c_y_2[3]], [c_z_2[0], c_z_2[3]], color_2)

    for i in range(4):
        ax.plot3D([c_x_1[i], c_x_1[-1]], [c_y_1[i], c_y_1[-1]], [c_z_1[i], c_z_1[-1]], color_1)
        ax.plot3D([c_x_2[i], c_x_2[-1]], [c_y_2[i], c_y_2[-1]], [c_z_2[i], c_z_2[-1]], color_2)

    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c='blue')

    set_axes_equal(ax)

    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(name)
