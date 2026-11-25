"""Relax movie: watch the glass relax instead of inferring it post-mortem.

RelaxMovie is a relax() callback (chainable with any other): every call it
renders the current ensemble as two scatter panels (xy and xz), coloured by
log10(rho/rho0) - the pseudo-pressure drive - so overpressure bubbles, edge
strays and the step-controller stall are all directly visible as they develop.
finish() assembles
the frames into an mp4 with ffmpeg.

Frame cadence is the relax's callback_every (the movie renders every call it
receives); frames accumulate across cascade stages/levels within one movie.
Usage:

    movie = RelaxMovie('run.mp4')
    relax(..., callback=movie, callback_every=10)     # or chain your own
    print(movie.finish())                             # -> path of the mp4
"""
import os
import shutil
import subprocess

import numpy as np

__all__ = ['RelaxMovie']


class RelaxMovie:
    """Frame-dumping relax callback + ffmpeg assembly.

    path       : output movie path ('.mp4' appended if no extension). Frames
                 are staged in '<path>.frames/' and removed after a successful
                 finish() unless keep_frames.
    max_points : per-frame scatter budget; larger ensembles are stride-
                 subsampled (deterministic, no flicker).
    clim       : colour range in DECADES of rho/rho0 (symmetric about 0).
    cmap       : matplotlib colormap for log10(rho/rho0). Default 'turbo'
                 (rainbow: blue = underdense, green = relaxed, red = over-
                 pressure). 'coolwarm' is a diverging render, whose
                 midpoint reads WHITE = relaxed - on a rainbow the converged
                 state is mid-spectrum instead, so judge convergence by the
                 colourbar, not by "washes out".
    fps        : movie frame rate.
    lim        : ((xlo,xhi),(ylo,yhi),(zlo,zhi)) axis limits; None = frozen
                 from the first frame's extent (all frames share axes so the
                 motion is real, not axis rescaling).
    every      : render every n-th callback invocation (on top of the relax's
                 own callback_every throttle; 1 = every call).
    """

    def __init__(self, path, max_points=60000, clim=2.0, fps=14, lim=None,
                 dpi=110, every=1, keep_frames=False, cmap='turbo'):
        base, ext = os.path.splitext(path)
        self.path = path if ext else path + '.mp4'
        self.framedir = base + '.frames'
        os.makedirs(self.framedir, exist_ok=True)
        for f in os.listdir(self.framedir):     # stale frames poison ffmpeg
            if f.endswith('.png'):
                os.unlink(os.path.join(self.framedir, f))
        self.max_points = int(max_points)
        self.clim = float(clim)
        self.cmap = str(cmap)
        self.fps = int(fps)
        self.lim = lim
        self.dpi = int(dpi)
        self.every = max(int(every), 1)
        self.keep_frames = keep_frames
        self.nframe = 0
        self.ncall = 0
        self.stage = 1
        self._last_it = None
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self._plt = plt

    def __call__(self, st):
        """relax() callback: st is an IterStats (pos/rho/rho0/it/...)."""
        if self._last_it is not None and st.it <= self._last_it:
            self.stage += 1                      # a new relax() started
        self._last_it = st.it
        self.ncall += 1
        if (self.ncall - 1) % self.every:
            return
        pos = st.pos
        step = max(1, int(np.ceil(len(pos) / self.max_points)))
        p = pos[::step]
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.log10(np.maximum(st.rho[::step], 1e-300)
                         / np.maximum(st.rho0[::step], 1e-300))
        if self.lim is None:                     # freeze axes on frame 1
            lo = p.min(0)
            hi = p.max(0)
            pad = 0.05 * (hi - lo)
            self.lim = tuple((lo[i] - pad[i], hi[i] + pad[i]) for i in range(3))
        plt = self._plt
        fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.4),
                                constrained_layout=True)
        for ax, (i, j), name in ((axs[0], (0, 1), 'x-y'),
                                 (axs[1], (0, 2), 'x-z')):
            sc = ax.scatter(p[:, i], p[:, j], c=c, s=0.7, linewidths=0,
                            cmap=self.cmap, vmin=-self.clim, vmax=self.clim,
                            rasterized=True)
            ax.set_xlim(*self.lim[i])
            ax.set_ylim(*self.lim[j])
            ax.set_aspect('equal')
            ax.set_title(name, fontsize=10)
        fig.colorbar(sc, ax=axs, shrink=0.85, label=r'log10 rho/rho0')
        fig.suptitle(f'stage {self.stage}  iter {st.it}  '
                     f'denserr {st.dens_err:.3f}  icr0 {st.icr0:.2e}  '
                     f'n {len(pos)}', fontsize=11)
        fig.savefig(os.path.join(self.framedir, f'f{self.nframe:05d}.png'),
                    dpi=self.dpi)
        plt.close(fig)
        self.nframe += 1

    def finish(self):
        """Assemble the frames into the mp4; returns its path (or the frame
        directory if ffmpeg is unavailable/fails or no frames were drawn)."""
        if self.nframe == 0:
            return self.framedir
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            return self.framedir
        # yuv420p needs even dims; scale filter rounds down to even
        cmd = [ffmpeg, '-y', '-loglevel', 'error',
               '-framerate', str(self.fps),
               '-i', os.path.join(self.framedir, 'f%05d.png'),
               '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
               '-pix_fmt', 'yuv420p', self.path]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            return self.framedir
        if not self.keep_frames:
            shutil.rmtree(self.framedir, ignore_errors=True)
        return self.path
