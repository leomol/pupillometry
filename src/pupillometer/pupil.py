# 2020-12-30. Leonardo Molina.
# 2021-04-30. Last modified.

import datetime
import json
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from colorsys import rgb_to_hsv
from flexible import Flexible
from matplotlib import animation
from matplotlib.widgets import PolygonSelector
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from threading import Lock
from videoCapture import VideoCapture
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
from matplotlib.cbook import mplDeprecation
filterwarnings("ignore", category=ConvergenceWarning)
filterwarnings("ignore", category=mplDeprecation)

class Pupil():
    @staticmethod
    def rgb2hsv(colors):
        return np.array([rgb_to_hsv(rgb[0], rgb[1], rgb[2]) for rgb in colors])
    
    @property
    def playing(self):
        p = self.__private
        return p.play and p.run
    
    @property
    def time(self):
        p = self.__private
        return p.stream.time
    
    @time.setter
    def time(self, value):
        p = self.__private
        p.stream.time = value
        if not self.playing:
            self.__onGrab()
    
    @property
    def index(self):
        p = self.__private
        return p.stream.index
    
    @index.setter
    def index(self, value):
        p = self.__private
        p.stream.index = value
        if not self.playing:
            self.__onGrab()
            
    def start(self):
        p = self.__private
        if not self.playing:
            # Hide polygon tool during playback.
            self.__setSelectorState(False)
            # Resume playback.
            p.play = True
            p.stream.start()
    
    def stop(self):
        p = self.__private
        if self.playing:
            p.stream.stop()
            # p.stream.join()
            if self.playing:
                p.play = False
    
    def __setSelectorState(self, state):
        p = self.__private
        p.roiSelector.set_visible(state)
        p.roiSelector.active = state
    
    def __onKeyPress(self, event):
        p = self.__private
        key = event.key
        
        # When exporting, the only possible action is to quit.
        if p.exporting and key != 'q':
            return
        
        # Playback controls.
        if key == ' ':
            if self.playing:
                self.stop()
            else:
                self.start()
        
        # Save and exit.
        elif key == 'q':
            self.__save()
            self.dispose()
        
        # Save.
        elif key == 's':
            self.__save()
        
        # Move thru time/frame.
        elif key == "home":
            self.index = 1
        elif key == "end":
            self.index = p.nFrames
        elif key == "right":
            self.index += 1
        if key == "left":
            self.index -= 1
        elif key == "up":
            self.time += 10
        elif key == "down":
            self.time -= 10
        elif key == "ctrl+right":
            self.time += 1
        elif key == "ctrl+left":
            self.time -= 1
            
        # Move thru entries.
        elif key == "ctrl+up":
            position = self.__getEntryPosition()
            if p.indices.size > position + 1:
                self.index = p.indices[position + 1]
        elif key == "ctrl+down":
            position = self.__getEntryPosition()
            if position >= 0 and self.index > p.indices[position]:
                self.index = p.indices[position]
            elif position > 0 and p.indices[position] == self.index:
                self.index = p.indices[position - 1]
                
        # Adding entries.
        elif key == 'a':
            self.stop()
            position = self.__entryAction("add")
            self.__process(position)
        elif key == 'd':
            self.stop()
            position = self.__entryAction("delete")
            self.__process(position)
        elif key == "enter":
            self.stop()
            position = self.__entryAction("commit")
            self.__process(position)
        
        # Changing parameter according to mode.
        elif key in ['+', '-']:
            position = self.__getEntryPosition()
            if position >= 0:
                self.stop()
                if p.modes[position] in ["cluster:accuracy", "cluster:speed"]:
                    step = 1 if key == '+' else -1
                    self.__changeDepth(position, step)
                    self.__process(position)
                elif p.modes[position] in ["dark", "bright"]:
                    step = 0.005 * (1 if key == '+' else -1)
                    self.__changeThreshold(position, step)
                    self.__postprocess(position)
                    self.__render(p.processed, position)
                    self.__drawROI(position)
        
        # Change delta.
        elif key in ['[', ']']:
            position = self.__getEntryPosition()
            if position >= 0:
                self.stop()
                step = 0.2 if key == ']' else -0.2
                self.__changeDelta(position, step)
                self.__postprocess(position)
                self.__render(p.processed, position)
                self.__drawROI(position)
        
        # Change erosion.
        elif key in ["ctrl+[", "ctrl+]"]:
            position = self.__getEntryPosition()
            if position >= 0:
                self.stop()
                step = 1 if key == "ctrl+]" else -1
                self.__changeErosion(position, step)
                self.__postprocess(position)
                self.__render(p.processed, position)
                self.__drawROI(position)
        
        # Cluster mode: change color.
        elif key == "tab":
            self.stop()
            position = self.__getEntryPosition()
            self.__changeColor(position)
            self.__postprocess(position)
            self.__render(p.processed, position)
            self.__drawROI(position)
        
        # Cluster mode: change view.
        elif key == 'z':
            p.showPosterization = not p.showPosterization
            position = self.__getEntryPosition()
            self.__process(position)
            
        # Change mode.
        elif key == 'm':
            position = self.__getEntryPosition()
            if position >= 0:
                if p.lastMode == "bright":
                    p.lastMode = "dark"
                elif p.lastMode == "dark":
                    p.lastMode = "cluster:accuracy"
                elif p.lastMode == "cluster:accuracy":
                    p.lastMode = "cluster:speed"
                elif p.lastMode == "cluster:speed":
                    p.lastMode = "bright"
                p.modes[position] = p.lastMode
                self.__process(position)
                
        # Export.
        elif key == "ctrl+enter":
            self.__export()
    
    def __export(self):
        p = self.__private
        nEntries = p.indices.size
        if nEntries == 0:
            print("[error] At least one ROI required.")
            success = False
        else:
            # Max width and height from all regions of interest.
            p.maxWidth = 0
            p.maxHeight = 0
            p.exporting = True
            for k in range(nEntries):
                contour = (p.vertices[k] + 0.5) * p.resolution
                contour = contour.round().astype(np.int32)
                width, height = contour.max(axis=0) - contour.min(axis=0)
                p.maxWidth = max(p.maxWidth, width)
                p.maxHeight = max(p.maxHeight, height)
            if p.videoWriter.open(
                filename=p.avi,
                fourcc=cv.VideoWriter_fourcc(*"FMP4"), # !! lossy
                fps=p.fps,
                frameSize=(p.maxWidth, p.maxHeight),
                isColor=True):
                    success = True
                    p.exporting = True
                    p.valid = np.full((p.nFrames, 1), False)
                    p.center = np.zeros((p.nFrames, 2), np.float)
                    p.axes = np.zeros((p.nFrames, 2), np.float)
                    p.angle = np.zeros((p.nFrames, 1), np.float)
                    p.moments = np.zeros((p.nFrames, 3), np.float)
                    # Process current frame.
                    self.__onGrab()
                    # Continue with stream.
                    self.start()
            else:
                print("[error] Cannot write output video file.")
                success = False
        return success
    
    def __changeThreshold(self, position, step):
        p = self.__private
        if position >= 0:
            # 0 <= threshold <= 1.
            p.thresholds[position] = min(1, max(0, p.thresholds[position] + step))

    def __changeErosion(self, position, step):
        p = self.__private
        if position >= 0:
            # 0 <= erosion.
            p.erosions[position] = max(p.erosions[position] + step, 0)

    def __changeDelta(self, position, step):
        p = self.__private
        if position >= 0:
            # -Inf < delta < Inf.
            p.deltas[position] += step
    
    def __changeColor(self, position):
        p = self.__private
        if position >= 0:
            # Cycle through color levels.
            lower = 0
            upper = p.depths[position] - 1
            value = p.colorIndex + 1
            if value > upper:
                value = lower
            elif value < lower:
                value = upper
            p.colorIndex = value

            p.colors[position] = p.qHSV[value]
            p.lastColor = p.colors[position]
    
    def __changeDepth(self, position, step):
        # Keep posterization within limits.
        p = self.__private
        if position >= 0:
            lower = 2
            upper = 20
            value = p.depths[position] + step
            p.depths[position] = min(max(value, lower), upper)
            p.lastDepth = p.depths[position]

    def __entryAction(self, action):
        # Add and remove ROIs.
        p = self.__private
        if action == "add":
            p.addedIndex = self.index
            self.__setSelectorState(True)
        elif action == "delete":
            nEntries = p.indices.size
            if nEntries > 0:
                position = nEntries - np.argmax(p.indices[::-1] <= self.index) - 1
                p.indices = np.delete(p.indices, position)
                del p.vertices[position]
                del p.modes[position]
                del p.depths[position]
                del p.colors[position]
                del p.thresholds[position]
                del p.erosions[position]
                del p.deltas[position]
            p.addedIndex = -1
            self.__setSelectorState(False)
        elif action == "commit":
            # Debug.
            if p.addedIndex == -2:
                vertices = p.lastVertices
                p.addedIndex = 0
            else:
                vertices = np.array(p.roiSelector.verts)
                
            if p.addedIndex >= 0:
                if vertices.size > 2:
                    if np.any(p.indices == p.addedIndex):
                        position = np.argmax(p.indices == p.addedIndex)
                        p.vertices[position] = vertices
                        p.modes[position] = p.lastMode
                        p.depths[position] = p.lastDepth
                        p.colors[position] = p.lastColor
                        p.thresholds[position] = p.lastThreshold
                        p.erosions[position] = p.lastErosion
                        p.deltas[position] = p.lastDelta
                    else:
                        nEntries = p.indices.size
                        position = 0 if nEntries == 0 else nEntries - np.argmax(p.indices[::-1] < p.addedIndex)
                        p.indices = np.insert(p.indices, position, p.addedIndex)
                        p.vertices.insert(position, vertices)
                        p.modes.insert(position, p.lastMode)
                        p.depths.insert(position, p.lastDepth)
                        p.colors.insert(position, p.lastColor)
                        p.thresholds.insert(position, p.lastThreshold)
                        p.erosions.insert(position, p.lastErosion)
                        p.deltas.insert(position, p.lastDelta)
                    p.addedIndex = -1
            self.__setSelectorState(False)
        position = self.__getEntryPosition()
        return position

    def __onGrab(self):
        # Retrieve image.
        p = self.__private
        if not p.run:
            return
        
        # Retrieve current frame and sleep according to caller.
        p.stream.retrieve(p.raw)

        # Convert from BGR (cv's default) to RGB (pyplot's default)
        cv.cvtColor(src=p.raw, dst=p.rgb, code=cv.COLOR_BGR2RGB)
        
        # Blur, scan for target, draw ROI, draw pupil.
        position = self.__getEntryPosition()
        self.__process(position)
    
    def __process(self, position):
        p = self.__private
        self.__preprocess(position)
        self.__postprocess(position)
        self.__render(p.processed, position)
        self.__drawROI(position)

        if p.closeWhenDone and self.index == p.nFrames:
            p.run = False
            self.__save()
            self.stop()
            p.stream.release()
    
    def __preprocess(self, position):
        p = self.__private
        p.processed = p.rgb.copy()
        if position >= 0:
            # Preprocess according to mode.
            if p.modes[position] in ["cluster:accuracy", "cluster:speed"]:
                # Get mask from ROI.
                mask = self.__getRoiMask(position)
                if np.any(mask):
                    depth = p.depths[position]
                    qLabels, qColors = self.__posterize(mask, depth, p.modes[position] == "cluster:accuracy")
                    # Posterize hsv and rgb (when requested).
                    [maskI, maskJ] = np.nonzero(mask)
                    for label in range(depth):
                        k = qLabels == label
                        if p.showPosterization:
                            p.processed[maskI[k], maskJ[k], :] = np.round(qColors[label, :]).astype(np.int)
                    p.qLabels = qLabels
                    p.qRGB = qColors
                    p.qHSV = Pupil.rgb2hsv(qColors)
            elif p.modes[position] == "bright":
                pass
            elif p.modes[position] == "dark":
                pass
    
    def __postprocess(self, position):
        p = self.__private
        if position >= 0:
            mask = self.__getRoiMask(position)
            if p.modes[position] in ["cluster:accuracy", "cluster:speed"]:
                # Closest color in current frame.
                closestLabel = np.argmin(np.sqrt(np.sum(np.power(p.qHSV - p.colors[position], 2), axis=1)))
                mask[mask > 0] = p.qLabels == closestLabel
            elif p.modes[position] == "bright":
                # cv.GaussianBlur(src=p.raw, dst=p.var, ksize=p.blurSizePixels, sigmaX=0)
                p.var = p.raw
                gray = cv.cvtColor(src=p.var, code=cv.COLOR_BGR2GRAY)
                threshold = int(p.thresholds[position] * 255)
                _, thresholded = cv.threshold(src=gray[mask > 0], thresh=threshold, maxval=255, type=cv.THRESH_BINARY)
                mask[mask > 0] = np.squeeze(thresholded)
            elif p.modes[position] == "dark":
                # cv.GaussianBlur(src=p.raw, dst=p.var, ksize=p.blurSizePixels, sigmaX=0)
                p.var = p.raw
                gray = cv.cvtColor(src=p.var, code=cv.COLOR_BGR2GRAY)
                threshold = int(p.thresholds[position] * 255)
                _, thresholded = cv.threshold(src=gray[mask > 0], thresh=threshold, maxval=255, type=cv.THRESH_BINARY_INV)
                mask[mask > 0] = np.squeeze(thresholded)
                
            # mask can be empty (e.g. no bright colors past threshold)
            if mask.sum() > 0:
                # Select largest blob, except background.
                [_, blobs, stats, centroids] = cv.connectedComponentsWithStats(image=mask)
                areas = stats[1:, 4]
                largest = np.argmax(areas) + 1
                mask = (blobs == largest).astype(np.uint8)
                
                # Erode/dilate.
                size = p.erosions[position]
                if size > 0:
                    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size + 1, size + 1))
                    mask = cv.erode(src=mask, kernel=kernel)
                    #mask = cv.morphologyEx(src=mask, op=cv.MORPH_OPEN, kernel=kernel)
            
                # # Remove extreme points.
                # pupilIJ = np.argwhere(mask)
                # [pupilI, pupilJ] = np.nonzero(mask)
                # ee = EllipticEnvelope(contamination=0.25, random_state=0)
                # labels = ee.fit_predict(pupilIJ)
                # labels = np.squeeze(labels)
                # mask[pupilI[labels == -1], pupilJ[labels == -1]] = 0
            
            m = cv.moments(mask)
            if m["m00"] == 0:
                # No points in mask.
                major, minor, uMajor, uMinor, angle = 0, 0, 0, 0, 0
                x, y = p.lastX, p.lastY
                ux, uy = x / p.width - 0.5, y / p.height - 0.5
                pupilXs, pupilYs = np.array(ux), np.array(uy)
            else:
                [pupilI, pupilJ] = np.nonzero(mask)
                pupilXs, pupilYs = pupilJ / p.width - 0.5, pupilI / p.height - 0.5
                
                [contours, hierarchy] = cv.findContours(
                    image=mask,
                    mode=cv.RETR_EXTERNAL,
                    method=cv.CHAIN_APPROX_NONE
                )
                # Option 1: Use perimeter
                # hull = cv.convexHull(contours[0])
                boundary = np.array(contours[0], dtype=np.float)[:, 0, :]
                
                # Option 2: Use whole body.
                # pupilIJ = np.argwhere(mask)
                # boundary = np.fliplr(pupilIJ).astype(np.float)

                if boundary.shape[0] >= 7 and p.ellipseModel.estimate(boundary):
                    # Ellipse model requires at least 5 vertices.
                    # Not all data can be successfully fit.
                    
                    # Fit ellipse in pixel coordinates.
                    x, y, major, minor, angle = p.ellipseModel.params
                    p.lastX, p.lastY = x, y
                    
                    # Fit ellipse in normalized coordinates.
                    uBoundary = boundary / p.resolution - 0.5
                    p.ellipseModel.estimate(uBoundary)
                    ux, uy, uMajor, uMinor, angle = p.ellipseModel.params
                else:
                    # Use pixel data.
                    angle = 0
                    major = minor = np.sqrt(m["m00"] / np.pi)
                    uMajor, uMinor = major / p.width, minor / p.height
                    x = m["m10"] / m["m00"]
                    y = m["m01"] / m["m00"]
                    ux, uy = x / p.width, y / p.height
                    p.lastX, p.lastY = x, y
            
                # Grow/shrink in pixels.
                major, minor = major + p.deltas[position], minor + p.deltas[position]
                # Grow/shrink in normalized units.
                uMajor, uMinor = uMajor + p.deltas[position] / p.width, uMinor + p.deltas[position] / p.height
                
            # Draw blob points.
            p.pupilPoints.set_xdata(pupilXs)
            p.pupilPoints.set_ydata(pupilYs)
            
            center = np.array([x, y])
            uCenter = np.array([ux, uy])
            axes = np.array([major, minor])

            # Draw ellipse.
            p.pupilEllipse.width = 2 * uMajor
            p.pupilEllipse.height = 2 * uMinor
            p.pupilEllipse.angle = angle / np.pi * 180
            p.pupilEllipse.set_center(uCenter)
        
        valid = position >= 0
        p.pupilEllipse.set_visible(valid)
        p.pupilPoints.set_visible(valid)
        p.roiPolygonInset.set_visible(valid)
        
        if p.exporting:
            # Update values for current frame.
            k = self.index - 1
            if valid:
                p.valid[k] = True
                p.center[k] = center
                p.axes[k] = axes
                p.angle[k] = angle
                p.moments[k] = np.array([m["m00"], m["m10"], m["m01"]])
            else:
                p.valid[k] = False
                position = 0
            
            # Get cropped video.
            contour = (p.vertices[position] + 0.5) * p.resolution
            contour = contour.round().astype(np.int32)
            j1, i1 = contour.min(axis=0)
            j2, i2 = contour.max(axis=0)
            di = i2 - i1
            dj = j2 - j1
            # Center and extend.
            dw = p.maxWidth - dj
            dh = p.maxHeight - di
            j1 -= dw // 2
            j2 += dw - dw // 2
            i1 -= dh // 2
            i2 += dh - dh // 2
            p.videoWriter.write(p.raw[i1:i2, j1:j2])
    
    def __posterize(self, mask, depth, accurate=True):
        p = self.__private
        # Posterize selection.
        cv.GaussianBlur(src=p.rgb, dst=p.var, ksize=p.blurSizePixels, sigmaX=0)
        rgbFeatures = p.var[mask > 0, :]
        rgbFeatures = rgbFeatures.reshape((-1, 3)).astype(np.float32)
        
        if accurate:
            model = GaussianMixture(
                n_components=depth,
                covariance_type="full",
                max_iter=2,
                warm_start=True
            )
            qLabels = model.fit_predict(rgbFeatures)
            qColors = model.means_
        else:
            [_, qLabels, qColors] = cv.kmeans(
                data=rgbFeatures,
                K=depth,
                bestLabels=None,
                criteria=p.kMeansCriteria,
                attempts=2,
                flags=cv.KMEANS_RANDOM_CENTERS
            )
            qLabels = np.squeeze(qLabels)
            
        return qLabels, qColors

    def __drawROI(self, position):
        # Draw ROI for current position.
        p = self.__private
        if position >= 0:
            if p.indices[position] == self.index:
                linestyle = '-'
                linewidth = 2
            else:
                linestyle = '--'
                linewidth = 1
            vertices = p.vertices[position]
            p.roiPolygonInset.set_xy(vertices)
            p.roiPolygonInset.set_linestyle(linestyle)
            p.roiPolygonInset.set_linewidth(linewidth)
            x1, x2, y1, y2 = min(vertices[:, 0]), max(vertices[:, 0]), min(vertices[:, 1]), max(vertices[:, 1])
            visible = True
        else:
            visible = False
        p.roiPolygonInset.set_visible(visible)
            
    def __getRoiMask(self, position):
        # Get mask from ROI.
        p = self.__private
        mask = np.zeros((p.height, p.width), np.uint8)
        roiContour = (p.vertices[position] + 0.5) * p.resolution
        roiContour = roiContour.astype(np.int32)
        cv.drawContours(
            image=mask,
            contours=[roiContour],
            contourIdx=-1,
            color=255,
            thickness=-1
        )
        return mask
    
    def __save(self):
        p = self.__private
        indices = p.indices.tolist()
        vertices = [value.tolist() for value in p.vertices]
        modes = p.modes
        depths = p.depths
        colors = [value.tolist() for value in p.colors]
        thresholds = p.thresholds
        erosions = p.erosions
        deltas = p.deltas
        entries = {
            "source": p.source,
            "frames": p.nFrames,
            "frame": self.index,
            "fps": p.fps,
            "width": p.width,
            "height": p.height,
            "indices": indices,
            "vertices": vertices,
            "modes": modes,
            "depths": depths,
            "colors": colors,
            "thresholds": thresholds,
            "erosions": erosions,
            "deltas": deltas
        }
        with open(p.json, 'w', encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=4)
        
        # Save whole csv file.
        if p.exporting:
            fmt = "%i", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f"
            time = np.arange(start=1, step=1, stop=p.nFrames + 1).reshape((-1, 1)) / p.fps
            data = np.hstack((p.valid, time, p.center, p.axes, p.angle, p.moments))
            np.savetxt(p.csv, data, fmt=fmt, delimiter=',', header="use, time, x, y, major, minor, angle, m00, m01, m10")
        
    def __getEntryPosition(self):
        # Find polygon applicable to current time.
        p = self.__private
        position = -1
        nEntries = p.indices.size
        if nEntries > 0:
            for k, frame in enumerate(p.indices):
                if frame <= self.index:
                    position = k
                else:
                    break
        return position

    def __onRender(self, *args):
        plt.draw()
        
    def __render(self, image, position=-1):
        p = self.__private
        
        image = image.copy()
        p.insetImage.set_data(image)
        
        if p.exporting:
            # Exporting message with progress.
            if self.index < p.nFrames:
                text = "[Exporting %i:%i ...]" % (self.index, p.nFrames)
            else:
                text = "[Finished exporting %i:%i]" % (self.index, p.nFrames)
        elif position == -1:
            # No ROI found.
            text = "[%.2f|%i:%i]" % (self.time, self.index, p.nFrames)
        else:
            # ROI present.
            mode = p.modes[position]
            delta = p.deltas[position]
            erosion = p.erosions[position]
            entry = position + 1
            nEntries = p.indices.size
            if mode in ["cluster:speed", "cluster:accuracy"]:
                depth = p.depths[position]
                text = "[%.2f|%i:%i] [ROI %i:%i] [Mode %s] [Depth %i] [Delta %.2f] [Erosion %i]" % (self.time, self.index, p.nFrames, entry, nEntries, mode, depth, delta, erosion)
            else:
                threshold = p.thresholds[position]
                text = "[%.2f|%i:%i] [ROI %i:%i] [Mode %s] [Threshold %.3f] [Delta %.2f] [Erosion %i]" % (self.time, self.index, p.nFrames, entry, nEntries, mode, threshold, delta, erosion)

        # Force aspect ratio.
        p.insetAxes.set_aspect(p.height / p.width)
        p.insetAxes.set_title(text)
        # !! if not p.exporting or self.index % 100 == 0 or self.index == p.nFrames:
        # plt.draw()

    def __onClose(self, event):
        p = self.__private
        self.stop()

    def dispose(self):
        p = self.__private
        # Dispose action may only happen once.
        if p.run:
            p.run = False
            self.stop()
            p.stream.release()
            p.stream.join()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.dispose()

    def __init__(self, source=r"C:\Users\Molina\Documents\public\MATLAB\pupil\tracker\amph01_vid_15fps.mp4", export=False):
        # Hide private members.
        p = self.__private = Flexible()

        # Setup asynchronous video capture.
        p.stream = VideoCapture()
        p.stream.subscribe("grab", lambda stream: self.__onGrab())

        # Show figure and wait for user interaction.
        if p.stream.open(source):
            p.nFrames = int(p.stream.get(cv.CAP_PROP_FRAME_COUNT))
            p.fps = int(p.stream.get(cv.CAP_PROP_FPS))
            p.resolution = np.array(p.stream.resolution) # width, height, 3
            p.width = p.resolution[0].item()
            p.height = p.resolution[1].item()
            
            # Identify playback session.
            p.source = source
            sourcePL = Path(source)
            p.json = "%s/%s-annotations.json" % (str(sourcePL.parent), sourcePL.stem)
            p.csv = "%s/%s-tracking.csv" % (str(sourcePL.parent), sourcePL.stem)
            p.avi = "%s/%s-cropped.avi" % (str(sourcePL.parent), sourcePL.stem)
            p.timestamp = "%s" % datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            path = Path(p.source)
            p.session = path.stem
        
            # Initialize reusable matrices. Numpy expects images dimensions as (height, width, 3).
            p.raw = np.zeros((p.height, p.width, 3), np.uint8)
            p.rgb = np.zeros((p.height, p.width, 3), np.uint8)
            p.var = np.zeros((p.height, p.width, 3), np.uint8)
            p.processed = np.zeros((p.height, p.width, 3), np.uint8)
            
            # Initialize playback state.
            p.play = False
            p.run = True
            p.lock = Lock()
        
            # Setup graphics.
            p.figure = plt.figure()
            p.figure.canvas.set_window_title(source)
            p.figure.canvas.mpl_connect("key_press_event", self.__onKeyPress)
            p.figure.canvas.mpl_connect("close_event", self.__onClose)
            # Image inset.
            p.insetAxes = plt.axes()
            p.insetAxes.set_axis_off()
            p.insetImage = p.insetAxes.imshow(np.zeros((1, 1, 3), np.uint8))
            
            # Initialize graphics.
            p.lastVertices = np.array([[ 0.07217742, -0.1916129 ], [-0.05201613, -0.24483871], [-0.00322581, -0.32763441], [ 0.05887097, -0.30989247]])
            p.roiPolygonInset = plt.Polygon(p.lastVertices, closed=True, fill=False, linewidth=1, color="#FF0000")
            p.pupilPoints = plt.plot([], [], linestyle='None', marker='s', color="#0000FF", alpha=0.15)[0]
            p.pupilEllipse = Ellipse((0, 0), 1, 1, 0, edgecolor="#00FF00", facecolor="none", linewidth=2)
            p.roiSelector = PolygonSelector(p.insetAxes, lambda vertices : None)
            p.ellipseModel = EllipseModel()
            
            self.__setSelectorState(False)
            p.insetAxes.add_patch(p.roiPolygonInset)
            p.insetAxes.add_patch(p.pupilEllipse)
            
            # Mode:
            #  bright: Brightest color down to a threshold.
            #  dark: Darkest color up to a threshold.
            #  cluster: Posterize and match an arbitrary color.
            #    cluster for accuracy: Clustering method is GaussianMixture - heterogeneous covariances.
            #    cluster for speed: Clustering method is k-means - homogeneous clusters.
            
            # Load saved session, if any.
            if Path(p.json).is_file():
                with open(p.json) as f:
                    entries = json.load(f)
                    frame = entries["frame"]
                    indices = entries["indices"]
                    vertices = entries["vertices"]
                    modes = entries["modes"]
                    depths = entries["depths"]
                    colors = entries["colors"]
                    thresholds = entries["thresholds"]
                    erosions = entries["erosions"]
                    deltas = entries["deltas"]
            else:
                frame = 1
                indices = []
                vertices = []
                modes = []
                depths = []
                colors = []
                thresholds = []
                erosions = []
                deltas = []
            p.indices = np.array(indices, dtype=np.uint32)
            p.vertices = [np.array(value) for value in vertices]
            p.modes = modes
            p.depths = depths
            p.colors = [np.array(value) for value in colors]
            p.thresholds = thresholds
            p.erosions = erosions
            p.deltas = deltas
            
            p.lastMode = "cluster:accuracy" if not modes else modes[-1]
            p.lastDepth = 5 if not depths else depths[-1]
            p.lastColor = Pupil.rgb2hsv([[255, 255, 255]]) if not p.colors else p.colors[-1]
            p.lastThreshold = 0.5 if not p.thresholds else p.thresholds[-1]
            p.lastErosion = 0 if not p.erosions else p.erosions[-1]
            p.lastDelta = 0 if not p.deltas else p.deltas[-1]
            p.lastX = 0
            p.lastY = 0
            
            # Parsing states.
            p.videoWriter = cv.VideoWriter()
            p.addedIndex = -1
            p.colorIndex = 0
            p.showPosterization = True
            p.qHSV = np.zeros((p.lastDepth, 3), np.uint8)
            
            # Constants.
            p.kMeansCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.75)
            p.blurSizePixels = (9, 9)
            
            # Animate in the main thread.
            renderFrequency = 10
            self.__animation = animation.FuncAnimation(
                fig=p.figure,
                func=self.__onRender,
                frames=1,
                interval=1000 / renderFrequency,
                repeat=True
            )
            
            # Run.
            plt.rcParams["keymap.all_axes"] = [item for item in plt.rcParams["keymap.all_axes"] if item != 'a']
            plt.rcParams["keymap.save"] = [item for item in plt.rcParams["keymap.save"] if item != 's']
            plt.rcParams["keymap.home"] = [item for item in plt.rcParams["keymap.home"] if item != "home"]
            plt.rcParams["keymap.back"] = [item for item in plt.rcParams["keymap.back"] if item not in ['c', "left", "backspace"]]
            plt.rcParams["keymap.forward"] = [item for item in plt.rcParams["keymap.forward"] if item != "right"]
            
            p.exporting = False
            p.closeWhenDone = False
            if export:
                # Export from the beginning.
                self.index = 1
                if self.__export():
                    p.closeWhenDone = True
                    p.stream.join()
            else:
                self.index = frame
                plt.show(block=True)
            
            # Release.
            p.stream.release()
            p.stream.join()
            
            if export:
                p.videoWriter.release()
        else:
            print("[error] Could not open video stream.")


if __name__ == "__main__":
    import glob
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--annotate", metavar="PATH", help="path to input video, e.g. \"folder/video.avi\"")
    group.add_argument("--export", metavar="PATH", help="path to input video(s) with pattern expansion, e.g. \"folder/video*.avi\"")
    parser.add_argument("--force", default=False, action="store_true", help="override existing tracking data.")
    args = parser.parse_args()
    
    if args.annotate:
        path = Path(args.annotate)
        source = path.as_posix()
        pp = Pupil(source, export=False)
    else:
        pattern = args.export
        filenames = glob.glob(pattern)
        for filename in filenames:
            file = Path(filename)
            if file.is_file():
                source = file.as_posix()
                stream = cv.VideoCapture()
                if stream.open(source):
                    stream.release()

                    csvPL = file.parent / ("%s-tracking.csv" % file.stem)
                    jsonPL = file.parent / ("%s-annotations.json" % file.stem)
                    if csvPL.is_file() and not args.force:
                        print("Skipping. Already exported \"%s\"" % file)
                    elif jsonPL.is_file():
                        print("Exporting \"%s\"" % file)
                        Pupil(source, export=True)
                        print("Exported!!!")
                    else:
                        print("Skipping. No annotations found in \"%s\"" % file)