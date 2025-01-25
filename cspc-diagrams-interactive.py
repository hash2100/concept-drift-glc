import numpy as np

import matplotlib
#matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.widgets as widgets
import matplotlib.cm as cm


def generate_points_inside_circle(x_c, y_c, radius, n_points):
    # generate points inside the circle
    # x_cs_in, y_cs_in = 0.2 + 0.15 * r_cs * np.cos(theta_cs), 0.5 + 0.15 * r_cs * np.sin(theta_cs)

    r_cs = 1. * np.sqrt(np.random.random(2 * n_points))
    theta_cs = np.random.random(2 * n_points) * 2 * np.pi

    x_cs, y_cs = x_c + radius * r_cs * np.cos(theta_cs), y_c + radius * r_cs * np.sin(theta_cs)

    # filter out points with x > 1
    mask = (x_cs < 1)
    x_cs, y_cs = x_cs[mask], y_cs[mask]

    return x_cs[:n_points], y_cs[:n_points]


def circle_equation(x, y, x_c, y_c, radius):
    return (x - x_c) ** 2 / radius ** 2 + (y - y_c) ** 2 / radius ** 2


def generate_points_outside_circle(x_c, y_c, radius, n_points):
    # generate points outside the region
    # x_cs_out, y_cs_out = np.random.random(n_points), np.random.random(n_points)
    x_cs, y_cs = np.random.random(2 * n_points), np.random.random(2 * n_points)
    mask = (circle_equation(x_cs, y_cs, x_c=x_c, y_c=y_c, radius=radius) > 1)
    x_cs, y_cs = x_cs[mask], y_cs[mask]
    return x_cs[:n_points], y_cs[:n_points]


def generate_simulated_circles_dataset(
        parameters=[
            {'x_c': 0.2, 'y_c': 0.5, 'radius': 0.15},
            {'x_c': 0.4, 'y_c': 0.5, 'radius': 0.2},
            {'x_c': 0.6, 'y_c': 0.5, 'radius': 0.25},
            {'x_c': 0.8, 'y_c': 0.5, 'radius': 0.3}
        ],
        generate_points_inside_circle=generate_points_inside_circle,
        generate_points_outside_circle=generate_points_outside_circle,
        n_points=12500):
    """
    Generate clearly-delimited drift set.
    """

    x_circles_generated, y_circles_generated = \
        np.empty((2 * n_points * len(parameters, ))), np.empty((2 * n_points * len(parameters, )))

    cls_circles_generated = np.empty((2 * n_points * len(parameters, )))

    for (i, param) in enumerate(parameters):
        x_cs_in, y_cs_in = generate_points_inside_circle(
            x_c=param['x_c'], y_c=param['y_c'], radius=param['radius'], n_points=n_points)
        x_cs_out, y_cs_out = generate_points_outside_circle(
            x_c=param['x_c'], y_c=param['y_c'], radius=param['radius'], n_points=n_points)

        # test that points are inside/outside the circle
        assert np.all(circle_equation(
            x_cs_in, y_cs_in, x_c=param['x_c'], y_c=param['y_c'], radius=param['radius']) < 1)
        assert np.all(circle_equation(
            x_cs_out, y_cs_out, x_c=param['x_c'], y_c=param['y_c'], radius=param['radius']) > 1)
        assert x_cs_in.shape == (n_points,)
        assert y_cs_in.shape == (n_points,)
        assert x_cs_out.shape == (n_points,)
        assert y_cs_out.shape == (n_points,)

        # intertwin the in & out points
        def intertwin(a, b):
            c = np.empty((a.size + b.size,), dtype=a.dtype)
            c[0::2] = a
            c[1::2] = b
            return c

        x_cs = intertwin(x_cs_in, x_cs_out)
        y_cs = intertwin(y_cs_in, y_cs_out)

        x_circles_generated[i * n_points * 2: (i + 1) * n_points * 2] = x_cs
        y_circles_generated[i * n_points * 2: (i + 1) * n_points * 2] = y_cs

        cls_circles_generated[i * n_points * 2: (i + 1) * n_points * 2: 2] = -1
        cls_circles_generated[i * n_points * 2 + 1: (i + 1) * n_points * 2 + 1: 2] = +1

    return x_circles_generated, y_circles_generated, cls_circles_generated


def compute_moment(func, y, epoch_step, epoch_width=None, cls=None):
    if epoch_width is None:
        epoch_width = epoch_step

    return np.stack([
        func(
            y[i * epoch_step: i * epoch_step + epoch_width, :-1]
            if cls is None
            else y[i * epoch_step: i * epoch_step + epoch_width, :-1][
                y[i * epoch_step: i * epoch_step + epoch_width, -1] == cls
                ],
            axis=0
        )
        for i in range((y.shape[0] - epoch_width + epoch_step) // epoch_step)
    ])


def compute_dataset(data_scdset, scd_epoch_width=1000, scd_epoch_step=1000):
    """
    Compute the mean for each epoch step.
    """

    # compute for both classes
    scd_means_all = compute_moment(np.mean, y=data_scdset, epoch_step=scd_epoch_step, epoch_width=scd_epoch_width,
                                   cls=None)
    scd_stds_all = compute_moment(np.std, y=data_scdset, epoch_step=scd_epoch_step, epoch_width=scd_epoch_width,
                                  cls=None)

    scd_means_cls1 = compute_moment(np.mean, y=data_scdset, epoch_step=scd_epoch_step, epoch_width=scd_epoch_width,
                                    cls=-1)
    scd_stds_cls1 = compute_moment(np.std, y=data_scdset, epoch_step=scd_epoch_step, epoch_width=scd_epoch_width,
                                   cls=-1)

    scd_means_cls2 = compute_moment(np.mean, y=data_scdset, epoch_step=scd_epoch_step, epoch_width=scd_epoch_width,
                                    cls=1)
    scd_stds_cls2 = compute_moment(np.std, y=data_scdset, epoch_step=scd_epoch_step, epoch_width=scd_epoch_width, cls=1)

    # take the i'th feature and make an array out of it, where columns (axis=1) are the mean and std
    # axis 0 are the features, axis 1 the epochs and axis 2 the mean and std
    stackable = lambda means, stds: np.stack(
        [np.stack([means[:, i], stds[:, i]], axis=1) for i in range(data_scdset.shape[1] - 1)]
    )
    scd_data_all = stackable(scd_means_all, scd_stds_all)
    scd_data_cls1 = stackable(scd_means_cls1, scd_stds_cls1)
    scd_data_cls2 = stackable(scd_means_cls2, scd_stds_cls2)

    return scd_data_all, scd_data_cls1, scd_data_cls2


def plot_set(x_cs, y_cs, cls):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(x_cs[cls == -1], y_cs[cls == -1], 'o', c='b', alpha=0.02)
    ax.plot(x_cs[cls == +1], y_cs[cls == +1], 'o', c='r', alpha=0.02)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.grid()
    plt.show()


class Player:
    def __init__(self, fig, ax, segments, colors, pos=(0.1, 0.94), **kwargs):
        self.i = 0
        self.min = 0
        self.max = len(segments[0]) - 2
        self.runs = True
        self.forwards = True
        self.frame_no = None
        self.ax = ax
        self.fig = fig
        self.segments = segments
        self.colors = colors
        self.points = []
        self.setup(pos)

        # Create the FuncAnimation
        self.ani = FuncAnimation(self.fig, self.update, frames=self.play(),
                                 init_func=self.init_func, **kwargs)

        self.markers = [(0, )]

    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.i >= self.min and self.i <= self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs = True
        self.ani.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.frame_no = None
        self.ani.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.frame_no = 1
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def mark(self, event=None):
        self.mark_pos()

    def onestep(self):
        if self.i >= self.min and self.i <= self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        # Create a LineCollection that will hold the line segments
        self.line_collections = [
            LineCollection(segment[:1], colors=self.colors[:1], linewidths=1)
            for segment in self.segments
        ]

        for ax, i in zip(self.ax, range(len(self.line_collections))):
            ax.add_collection(self.line_collections[i])

            point, = ax.plot([], [], "o", color=[0, 0, 0, 1], alpha=0.6, ms=0.8)
            self.points.append(point)

        # Create buttons
        self.button_oneback = widgets.Button(self.fig.add_axes([pos[0] + 0.05, pos[1], 0.1, 0.04]),
                                             label='$\u29CF$') # Step back
        self.button_back = widgets.Button(self.fig.add_axes([pos[0] + 0.15, pos[1], 0.1, 0.04]),
                                          label='$\u25C0$')  # Back
        self.button_stop = widgets.Button(self.fig.add_axes([pos[0] + 0.25, pos[1], 0.1, 0.04]),
                                          label='$\u25A0$')  # Stop
        self.button_forward = widgets.Button(self.fig.add_axes([pos[0] + 0.35, pos[1], 0.1, 0.04]),
                                             label='$\u25B6$')  # Forward
        self.button_oneforward = widgets.Button(self.fig.add_axes([pos[0] + 0.45, pos[1], 0.1, 0.04]),
                                                label='$\u29D0$')  # Step forward
        self.button_mark = widgets.Button(self.fig.add_axes([pos[0] + 0.55, pos[1], 0.1, 0.04]),
                                                label='$\u29C6$')  # Mark

        # Button click events
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.button_mark.on_clicked(self.mark)

        # Slider setup
        sliderax = self.fig.add_axes([pos[0], pos[1] - 0.05, 0.64, 0.04])
        self.slider = widgets.Slider(sliderax, '', self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self, i):
        self.i = int(self.slider.val)
        self.slider.val = self.i
        self.slider.valtext.set_text('{}'.format(self.i))
        self.refresh(self.i)

    def mark_pos(self):
        pos = int(self.slider.val)

        # remove pair if it contains the pos
        k = [k for k in range(len(self.markers)) if pos in self.markers[k]]
        if len(k) > 0:
            k = k[0]
            self.markers = self.markers[:k] + self.markers[k + 1:]

            if len(self.markers) == 0:
                self.markers = [(0,)]
        else:
            # update markers pairs list
            if len(self.markers[-1]) == 1:
                self.markers[-1] = (self.markers[-1][0], pos)
            else:
                self.markers.append((pos, ))

        self.refresh(self.i)

        print(f"markers: {self.markers}")

    def init_func(self):
        return self.line_collections

    def update(self, i):
        self.slider.set_val(i)

        return self.refresh(i)

    def refresh(self, i):
        # Update the LineCollection by adding new segments progressively
        for line_collection, segment in zip(self.line_collections, self.segments):
            line_collection.set_paths(segment[:i + 1])
            line_collection.set_color(self.colors[:i + 1])  # Apply colors to the segments

        for k, point in enumerate(self.points):
            point.set_data(
                [item[0][0] for item in self.segments[k][:i + 1]],
                [item[0][1] for item in self.segments[k][:i + 1]]
            )

        for marker in self.markers if len(self.markers[-1]) == 2 else self.markers[:-1]:
            for f, segment in enumerate(self.segments):
                # compute the limits
                (min_x, max_x, min_y, max_y) = (
                    min(item[0][0] for item in segment[marker[0]:marker[1] + 1]),
                    max(item[0][0] for item in segment[marker[0]:marker[1] + 1]),
                    min(item[0][1] for item in segment[marker[0]:marker[1] + 1]),
                    max(item[0][1] for item in segment[marker[0]:marker[1] + 1])
                )
                self.ax[f].plot(
                    [min_x, max_x, max_x, min_x, min_x],
                    [min_y, min_y, max_y, max_y, min_y],
                    ls=':', color="blue", lw=0.6
                )
            prev_marker = marker

        if self.frame_no is not None and self.frame_no != i:
            self.frame_no = i
            self.fig.savefig(f"/Users/honorius/figs/figure-{self.frame_no:04d}.png")

        return self.line_collections


def generate_four_dim_dataset():
    x_circles_generated, y_circles_generated, cls_circles_generated = generate_simulated_circles_dataset()

    no_points = 12500
    epochs = 4
    length = no_points * 2 * epochs

    four_dim_dataset = np.random.rand(length, 4)

    four_dim_dataset[0: no_points * 2, 0] = x_circles_generated[0: no_points * 2]
    four_dim_dataset[0: no_points * 2, 1] = y_circles_generated[0: no_points * 2]

    four_dim_dataset[no_points * 2: no_points * 4, 1] = x_circles_generated[no_points * 2: no_points * 4]
    four_dim_dataset[no_points * 2: no_points * 4, 2] = y_circles_generated[no_points * 2: no_points * 4]

    four_dim_dataset[no_points * 4: no_points * 6, 2] = x_circles_generated[no_points * 4: no_points * 6]
    four_dim_dataset[no_points * 4: no_points * 6, 3] = y_circles_generated[no_points * 4: no_points * 6]

    four_dim_dataset[no_points * 6: no_points * 8, 0] = x_circles_generated[no_points * 6: no_points * 8]
    four_dim_dataset[no_points * 6: no_points * 8, 3] = y_circles_generated[no_points * 6: no_points * 8]

    data_4dset = np.concatenate([
        four_dim_dataset[:, 0].reshape(-1, 1),
        four_dim_dataset[:, 1].reshape(-1, 1),
        four_dim_dataset[:, 2].reshape(-1, 1),
        four_dim_dataset[:, 3].reshape(-1, 1),
        cls_circles_generated.reshape(-1, 1)
    ], axis=1)

    return data_4dset


def marking_cspc_diagram(data):
    # prepare information about the data
    n_feat = data.shape[0]
    n = data.shape[1]

    # Create pairs of adjacent points to form line segments
    segments = [
        [
            [(data[f, i, 0], data[f, i, 1]), (data[f, i + 1, 0], data[f, i + 1, 1])]
            for i in range(n - 1)
        ] for f in range(n_feat)
    ]

    # Generate a list of colors using a colormap
    colors = cm.copper(np.linspace(0, 1, n - 1))  # Change 'viridis' to any other colormap

    # on each line of the plot we put maximum two features
    lines = n_feat // 2 + (n_feat - n_feat // 2 * 2)

    # Create the plot
    fig = plt.figure(figsize=(11, 5 * lines + 1))

    ax = []
    for f in range(n_feat):
        ax.append(plt.subplot2grid(shape=(8 * lines + 2, 2), loc=((f // 2) * 5, f % 2), rowspan=4, colspan=1))

    for f, axe in enumerate(ax):
        axe.set_xlim([
            min(data[f, i, 0] for i in range(n)),
            max(data[f, i, 0] for i in range(n)),
        ])
        axe.set_ylim([
            min(data[f, i, 1] for i in range(n)),
            max(data[f, i, 1] for i in range(n)),
        ])

        axe.set_xlabel(f'mean(feature({f}))')
        axe.set_ylabel(f'std(feature({f}))')
        axe.grid()

    # Create the Player object (animation controller)
    player = Player(fig, ax, segments, colors, interval=25)

    # Show the plot
    plt.show()


def compute_sine1_dataset(epoch_width=1000, epoch_step=1000):
    import torch

    data = torch.load('./data/sine1.pt')
    # print(f"x shape: {data['x'].shape}")
    # print(f"y shape: {data['y'].shape}")

    # construct dataframe with all attributes as colums
    y = np.concatenate([data['x'].numpy(), data['y'].numpy().reshape(-1, 1)], axis=1)

    return compute_dataset(data_scdset=y, scd_epoch_width=epoch_width, scd_epoch_step=epoch_step)

def compute_covertype_dataset(epoch_width=1000, epoch_step=1000):
    import torch

    data = torch.load('./data/covertype.pt')
    # print(f"x shape: {data['x'].shape}")
    # print(f"y shape: {data['y'].shape}")

    # construct dataframe with all attributes as colums
    z = np.concatenate([data['x'].numpy(), data['y'].numpy().reshape(-1, 1)], axis=1)

    # select features with significant variation
    y = z[:, [0, 1, 2, 3, 5, 9, -1]]

    return compute_dataset(data_scdset=y, scd_epoch_width=epoch_width, scd_epoch_step=epoch_step)


def main():
    # x_circles_generated, y_circles_generated, cls_circles_generated = generate_simulated_circles_dataset()

    # plot_set(x_circles_generated, y_circles_generated, cls_circles_generated)

    # construct dataframe with all attributes as colums
    # data_scdset = np.concatenate([
    #     x_circles_generated.reshape(-1, 1),
    #     y_circles_generated.reshape(-1, 1),
    #     cls_circles_generated.reshape(-1, 1)
    # ], axis=1)
    #
    # print(f"shape: {data_scdset.shape}")

    # scd_data_all_100, scd_data_cls1_100, scd_data_cls2_100 = compute_dataset(data_scdset, scd_epoch_step=100)

    # marking_cspc_diagram(scd_data_all_100)
    # marking_cspc_diagram(scd_data_cls1_100)

    data_4dset = generate_four_dim_dataset()
    data_4d_all, data_4d_cls1, data_4d_cls2 = compute_dataset(data_4dset, scd_epoch_step=100)

    marking_cspc_diagram(data_4d_all)

    # $ convert -delay 5 -loop 0 *.png sine1-anim.gif

    # SINE1 dataset
    # scd_data_all_100, scd_data_cls1_100, scd_data_cls2_100 = compute_sine1_dataset(epoch_width=1000, epoch_step=100)
    # marking_cspc_diagram(scd_data_cls2_100)

    # COVERTYPE dataset
    # scd_data_all_250, scd_data_cls1_250, scd_data_cls2_250 = compute_covertype_dataset(epoch_width=1000, epoch_step=250)
    # marking_cspc_diagram(scd_data_cls2_250)



if __name__ == "__main__":
    main()
