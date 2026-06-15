import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no Tk window needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from pprint import pprint

class IberWoodResults:
    def __init__(self, iber_folder):
        self.IBER_FOLDER = iber_folder
        
        print(f"Iber Model: {self.IBER_FOLDER}")
        
    def load_wood_data(self):
        self.wood = pd.read_csv(
            self.IBER_FOLDER + "/Wood/Wood.rep",
            sep=';',
            comment='#',
            skip_blank_lines=True,
            skipinitialspace=True
        )

        # Keep only fully numeric rows
        self.wood = self.wood[pd.to_numeric(self.wood["Time(s)"], errors='coerce').notnull()]

        for col in self.wood.columns:
            self.wood[col] = pd.to_numeric(self.wood[col])

        self.wood.columns = [c.strip() for c in self.wood.columns]
        self.wood["Time(s)"] = self.wood["Time(s)"].astype(float)

        print(self.wood)

    def load_hydraulic_data(self, hydraulic_variable):

        #save hydraulic variable for class instance
        self.hydraulic_variable = hydraulic_variable
            
        #######################################
        # Helper functions
        #######################################

        def _extract_time(filename):
            name  = filename.replace(".asc", "")
            parts = name.split("_")
            for part in reversed(parts):
                try:
                    return float(part)
                except ValueError:
                    continue
            return None

        def _read_ascii_raster(filepath):
            with open(filepath, 'r') as f:

                ncols    = int(f.readline().split()[1])
                nrows    = int(f.readline().split()[1])

                # Header can be XLLCENTER or XLLCORNER — same for YLL
                xll_line = f.readline().split()
                yll_line = f.readline().split()
                xll_type = xll_line[0].upper()
                yll_type = yll_line[0].upper()
                xll      = float(xll_line[1])
                yll      = float(yll_line[1])

                cellsize = float(f.readline().split()[1])
                nodata   = float(f.readline().split()[1])
                data     = np.loadtxt(f)

            data[data == nodata] = np.nan

            # CORNER coords refer to the cell edge; shift by half a cell to get cell centres
            if "CORNER" in xll_type:
                xll += cellsize / 2
            if "CORNER" in yll_type:
                yll += cellsize / 2

            # ASC files store row 0 at the top (north); flip so row 0 is at the bottom
            # to match imshow origin='lower' — xll/yll remain the SW corner
            data = np.flipud(data)

            return data, ncols, nrows, xll, yll, cellsize
        
        #######################################
        # Organize the files
        #######################################
        self.raster_files = sorted(
            glob.glob(os.path.join(self.IBER_FOLDER + "/Rasters/Hydraulic", f"{hydraulic_variable}*.asc"))
        )

        pprint(self.raster_files)

        ########################################
        # Load raster data into python 
        ########################################

        self.rasters = []
        for file in self.raster_files:     
            time = _extract_time(os.path.basename(file))
            data, ncols, nrows, xll, yll, cellsize = _read_ascii_raster(file)
            self.rasters.append({"time": time, "data": data})

            #for the first raster read, save the extent details in the class instance
            if len(self.raster_files) == 0:
                self.ncols = ncols
                self.nrows = nrows
                self.xll = xll
                self.yll = yll
                self.cellsize = cellsize

        self.rasters = sorted(self.rasters, key=lambda x: x["time"])

        print(f"{hydraulic_variable} rasters loaded from files")

    def check_data_alignment(self):
        print("\n--- RASTER EXTENT ---")
        print(f"  XLLCENTER : {self.xll}")
        print(f"  YLLCENTER : {self.yll}")
        print(f"  cellsize  : {self.cellsize}")
        print(f"  ncols     : {self.ncols},  nrows: {self.nrows}")
        xmin_edge = self.xll - self.cellsize / 2
        xmax_edge = self.xll + (self.ncols - 0.5) * self.cellsize
        ymin_edge = self.yll - self.cellsize / 2
        ymax_edge = self.yll + (self.nrows - 0.5) * self.cellsize
        print(f"  x extent  : {xmin_edge:.4f}  →  {xmax_edge:.4f}")
        print(f"  y extent  : {ymin_edge:.4f}  →  {ymax_edge:.4f}")

        print("\n--- WOOD COORDINATE RANGE ---")
        print(f"  X : {self.wood['X'].min():.4f}  →  {self.wood['X'].max():.4f}")
        print(f"  Y : {self.wood['Y'].min():.4f}  →  {self.wood['Y'].max():.4f}")
        print()
    
    def save_results_as_video(self, CMAP, V_MIN, V_MAX, DPI, FPS, OUTPUT_FN):
        # ============================================================
        # SETUP FIGURE  (created once, reused every frame)
        # ============================================================

        fig, ax = plt.subplots(figsize=(14, 6))

        # xll/yll are cell centres (after any CORNER→CENTRE correction above)
        # imshow extent wants the outer edges of the border cells
        xmin   = self.xll - self.cellsize / 2
        ymin   = self.yll - self.cellsize / 2
        xmax   = self.xll + (self.ncols - 0.5) * self.cellsize
        ymax   = self.yll + (self.nrows - 0.5) * self.cellsize
        extent = [xmin, xmax, ymin, ymax]

        img = ax.imshow(
            self.rasters[0]["data"],
            extent=extent,
            origin='lower',
            cmap=CMAP,
            vmin=V_MIN,
            vmax=V_MAX
        )

        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label(self.hyraulic_variable)
        title = ax.set_title(f"Wood transport and {self.hydraulic_variable}")
        ax.set_aspect('equal')

        # ============================================================
        # DRAW WOOD PIECES
        # ============================================================

        wood_patches = []

        def _draw_wood(frame_time):
            global wood_patches
            for p in wood_patches:
                p.remove()
            wood_patches = []

            tol     = 0.01
            current = self.wood[np.abs(self.wood["Time(s)"] - frame_time) < tol]

            for _, row in current.iterrows():
                x         = row["X"]
                y         = row["Y"]
                length    = row["Length(m)"]
                diameter  = row["Diameter(m)"]
                angle_deg = np.degrees(row["Angle"])

                # Place rectangle centred on (x, y) with zero rotation,
                # then rotate about (x, y) via the transform — only one rotation applied
                rect = patches.Rectangle(
                    (x - length / 2, y - diameter / 2),
                    length,
                    diameter,
                    angle=0,                  # no built-in rotation
                    linewidth=1,
                    edgecolor='black',
                    facecolor='saddlebrown',
                    zorder=10
                )

                t = (
                    patches.transforms.Affine2D()
                    .rotate_deg_around(x, y, angle_deg)
                    + ax.transData
                )
                rect.set_transform(t)
                ax.add_patch(rect)
                wood_patches.append(rect)

        # ============================================================
        # RENDER ONE FRAME → numpy array (H, W, 3)  BGR for cv2
        # ============================================================

        def _render_frame(raster):
            img.set_array(raster["data"])
            title.set_text(f"Time = {raster['time']:.2f} s")
            _draw_wood(raster["time"])

            fig.canvas.draw()

            # buffer_rgba() returns RGBA bytes; works across all modern matplotlib versions
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            frame_rgba = buf.reshape(h, w, 4)

            # cv2 expects BGR (drop the alpha channel)
            return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
        
        # ============================================================
        # INITIALISE cv2.VideoWriter
        # ============================================================

        # Render frame 0 first so we know the exact pixel dimensions
        fig.canvas.draw()          # make sure the canvas is sized at DPI
        plt.tight_layout()
        fig.set_dpi(DPI)
        fig.canvas.draw()

        first_frame = _render_frame(self.rasters[0])
        frame_h, frame_w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # H.264-compatible container
        writer = cv2.VideoWriter(OUTPUT_FN, fourcc, FPS, (frame_w, frame_h))

        # Write the first frame we already rendered
        writer.write(first_frame)
        print(f"Frame 1 / {len(self.rasters)}")

        # ============================================================
        # RENDER REMAINING FRAMES
        # ============================================================

        for i, raster in enumerate(self.rasters[1:], start=2):
            frame = _render_frame(raster)
            writer.write(frame)
            print(f"Frame {i} / {len(self.rasters)}")

        # ============================================================
        # FINALISE
        # ============================================================

        writer.release()
        plt.close(fig)

        print("Animation saved:", OUTPUT_FN)


if __name__ == "__main__":
    results = IberWoodResults(iber_folder="c:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/first steps setup and calibration/test_0.25/TIFfp00875_ch01_rfsd025.gid")

    results.load_wood_data()

    results.load_hydraulic_data("Depth")

    results.check_data_alignment()
    
    results.save_results_as_video(CMAP ="Blues", V_MIN=0, V_MAX=0.6, DPI=200, FPS=10, 
                                  OUTPUT_FN= "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Playing around with the model/wood_flow_animation.mp4")


    
