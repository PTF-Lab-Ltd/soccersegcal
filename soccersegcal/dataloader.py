# Keep all original imports
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize, hflip
import torch
import json
from sncalib.soccerpitch import SoccerPitch
from vi3o.image import imviewsc, imview, imread, imscale
from vi3o import viewsc, view, flipp
import numpy as np
import cv2
from shapely.geometry import LineString, Point
from shapely.ops import split
from sncalib.detect_extremities import join_points

# Keep original Line class exactly as is


class Line:
    def __init__(self, pkts, image_shape):
        self.image_shape = image_shape
        self.missing = False
        pkts = [np.array((p['x'] * image_shape[1], p['y']
                         * image_shape[0])) for p in pkts]
        pkts = np.array(join_points(pkts, float('Inf'))[0])
        self.original = LineString(pkts)
        p1 = self.connect_to_image_border(pkts[0], self.direction(pkts))
        p2 = self.connect_to_image_border(pkts[-1], self.direction(pkts[::-1]))
        self.line_string = LineString(np.vstack([[p1], pkts, [p2]]))
        self.padding = [LineString([p1, pkts[0]]), LineString([pkts[-1], p2])]

    def direction(self, pkts):
        i = 1
        while i < len(pkts) - 1 and np.linalg.norm(pkts[0] - pkts[i]) < 10:
            i += 1
        return pkts[0] - pkts[i]

    def direction_at(self, pkts, pos):
        if np.linalg.norm(pkts[-1] - pos) < np.linalg.norm(pkts[0] - pos):
            pkts = pkts[::-1]
        return self.direction(pkts)

    def connect(self, other):
        if other.missing:
            return
        if self.line_string.intersects(other.line_string):
            self_parts = split(self.line_string, other.line_string)
            other_parts = split(other.line_string, self.line_string)
            pos = self.line_string.intersection(other.line_string)
            x, y = np.array(pos.coords)[0]
            if not (0 <= x < self.image_shape[1] and 0 <= y < self.image_shape[1]):
                return

            alternatives = []
            for p1 in self_parts.geoms:
                for p2 in other_parts.geoms:
                    alternatives.append((p1, p2))
            best = max(alternatives, key=lambda a: min(a[0].hausdorff_distance(
                pad) for pad in self.padding) + min(a[1].hausdorff_distance(pad) for pad in other.padding))
            self.line_string, other.line_string = best

    def connect_to_image_border(self, p, v):
        if p is None:
            return 0, None
        v = v / np.linalg.norm(v)
        tt = [-p[0] / v[0], -p[1] / v[1],
              (self.image_shape[1] - 1 - p[0]) / v[0], (self.image_shape[0] - 1 - p[1]) / v[1]]
        tt = [t for t in tt if t >= 0 and np.isfinite(t)]
        if len(tt) == 0:
            return p
        t = min(tt)
        return p + t * v

# Keep original MisingLine class exactly as is


class MisingLine(Line):
    def __init__(self, image_shape) -> None:
        self.image_shape = image_shape
        self.pkts = [None, None]
        self.v1 = self.v2 = None
        self.missing = True

    def connect(self, other):
        pass

    def match_direction(self, other):
        pass


class SoccerNetFieldSegmentationDataset(Dataset):
    def __init__(self, datasetpath="/home/data/soccernet_processed/01_frames/england_epl-2014-2015-2015-02-21 - 18-00 Chelsea 1 - 1 Burnley_0001", split="valid", width=640, height=None, skip_bad=False):
        if height is None:
            height = (width * 9) // 16
        self.root = Path(datasetpath)
        self.images = list((self.root).glob('*.jpg'))
        if not len(self.images):
            raise FileNotFoundError(
                f'No .jpg images found in {self.root / split}')
        self.images.sort()
        self.shape = (height, width)
        self.bad = {}  # {
        #     "valid": {11, 32, 60, 64, 71},  # truncated for brevity
        #     "test": {23, 33, 75},  # truncated for brevity
        #     "train": {81, 87, 89},  # truncated for brevity
        #     "challenge": set(),
        # }[split]

        self.indexes = list(range(len(self.images)))
        if skip_bad:
            self.indexes = [i for i in self.indexes if i not in self.bad]
            self.images = [img for i, img in enumerate(
                self.images) if i not in self.bad]
        self.class_names = ['FullField', 'CircleCentral',
                            'BigRect', 'CircleSide', 'SmallRect', 'Goal']
        self.split = split

    def lines(self, index):
        fn = self.images[index]
        file = fn.parent / fn.name.replace('.jpg', '.json')
        if not file.exists():
            return None
        with file.open() as fp:
            lines = json.load(fp)
        return {n: l for n, l in lines.items() if len(l) > 1}

    def show_lines(self, index, pause=True):
        # Keep original method
        img = imread(self.images[index])
        img = imscale(img, (self.shape[1], self.shape[0]))
        for n, points in self.lines(index).items():
            print(n)
            pkts = [np.array((p['x'] * img.shape[1], p['y'] * img.shape[0]))
                    for p in points]
            pkts = np.array(join_points(pkts, float('Inf'))[0])
            if 'Goal left' in n or ('Goal' not in n and 'left' in n):
                c1, c2 = 255, 0
            elif 'Goal right' in n or ('Goal' not in n and 'right' in n):
                c1, c2 = 0, 255
            else:
                c1, c2 = 128, 128
            if 'Side' in n:
                c3 = 128
            else:
                c3 = 0
            cv2.polylines(img, [np.int32(pkts)], False, (c1, c2, c3), 3)
        view(img, pause=pause)

    def save_lines(self, index, output_dir):
        """Save line visualization to disk"""
        img = imread(self.images[index])
        img = imscale(img, (self.shape[1], self.shape[0]))
        lines_data = self.lines(index)
        if lines_data is None:
            return

        for n, points in lines_data.items():
            pkts = [np.array((p['x'] * img.shape[1], p['y'] * img.shape[0]))
                    for p in points]
            pkts = np.array(join_points(pkts, float('Inf'))[0])
            if 'Goal left' in n or ('Goal' not in n and 'left' in n):
                c1, c2 = 255, 0
            elif 'Goal right' in n or ('Goal' not in n and 'right' in n):
                c1, c2 = 0, 255
            else:
                c1, c2 = 128, 128
            if 'Side' in n:
                c3 = 128
            else:
                c3 = 0
            cv2.polylines(img, [np.int32(pkts)], False, (c1, c2, c3), 3)

        # Save the image
        output_path = Path(output_dir) / f'{self.images[index].stem}_lines.jpg'
        cv2.imwrite(str(output_path), img)

    def save_visualization(self, entry, index, output_dir):
        """Save segmentation visualization to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save original image
        img_array = (entry['image'].cpu().numpy() *
                     255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imwrite(str(output_dir / f'{self.images[index].stem}_original.jpg'),
                    cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        # Save segmentation visualization if available
        if 'segments' in entry:
            segs = entry['segments'].cpu().numpy()
            seg_viz = np.zeros((*self.shape, 3), dtype=np.uint8)
            seg_viz[..., 0] = ((segs[:3].sum(0) > 0) * 128).astype(np.uint8)
            seg_viz[..., 1] = ((segs[3:].sum(0) > 0) * 128).astype(np.uint8)
            seg_viz[..., 2] = ((segs.sum(0) > 0) * 128).astype(np.uint8)
            cv2.imwrite(
                str(output_dir / f'{self.images[index].stem}_segments.jpg'), seg_viz)

    # Keep original __getitem__ exactly as is
    def __getitem__(self, index):
        fn = self.images[index]
        img = read_image(str(fn), ImageReadMode.RGB)
        img = resize(img, self.shape)
        lines = self.lines(index)

        dt = torch.get_default_dtype()
        if lines is None:
            return dict(
                image=img.to(dt).div(255),
                name=fn.name,
            )

        segments = np.zeros((6,) + self.shape, np.uint8)
        for name, area in SoccerPitch.field_areas.items():
            all_lines = [Line(lines[n], self.shape).original.coords
                         for n in area['contains'] if n in lines]
            if len(all_lines) == 0:
                continue
            center_x, center_y = map(
                int, np.round(np.vstack(all_lines).mean(0)))
            center_x = min(max(center_x, 0), self.shape[1] - 1)
            center_y = min(max(center_y, 0), self.shape[0] - 1)

            rim = []
            for n in area['border']:
                if n in lines:
                    rim.append(Line(lines[n], self.shape))
                else:
                    rim.append(MisingLine(self.shape))

            segs = np.zeros(self.shape, np.uint8)
            plot_lines = [np.int32(l.line_string.coords)
                          for l in rim if not l.missing]
            cv2.polylines(segs, plot_lines, False, 255, 1)

            if segs[max(center_y-1, 0):min(center_y+2, self.shape[0]), max(center_x-1, 0):min(center_x+2, self.shape[1])].max() == 255:
                for n in area['border']:
                    if n not in area['contains'] and n in lines:
                        l = Line(lines[n], self.shape).line_string
                        p = np.array(l.interpolate(
                            l.project(Point([center_x, center_y]))).coords)[0]
                        p -= (center_x, center_y)
                        l = np.linalg.norm(p)
                        if l > 0:
                            p /= l
                            center_x, center_y = map(
                                int, (center_x, center_y) + 2 * p)
                            center_x = min(max(center_x, 0), self.shape[1] - 1)
                            center_y = min(max(center_y, 0), self.shape[0] - 1)

            cv2.floodFill(segs, None, (center_x, center_y), 255)
            cv2.polylines(segs, plot_lines, False, 0, 1)
            segments[area['index']][segs > 0] = segs[segs > 0]

        return dict(
            image=img.to(dt).div(255),
            segments=torch.tensor(segments).to(dt).div(255),
            name=fn.name,
        )


def pview(img, pause=True):
    """Keep original pview function"""
    img = img.detach().cpu().numpy()
    if len(img.shape) == 3:
        img = img.transpose(1, 2, 0)
    viewsc(img, pause=pause)


def save_seg_vis(img, seg, output_path):
    """New function to save instead of view"""
    img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)

    seg = seg.detach().cpu().numpy()
    seg = seg.transpose(1, 2, 0)
    seg = (seg * 255).astype(np.uint8)

    is_black = np.all(seg < 20, axis=2)

    result = img.copy()

    alpha = 0.25
    for i in range(3):
        result[:, :, i] = np.where(is_black,
                                   img[:, :, i],
                                   img[:, :, i] * (1-alpha) + seg[:, :, i] * alpha)

    cv2.imwrite(str(output_path), cv2.cvtColor(
        result.astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    data = SoccerNetFieldSegmentationDataset(split='valid')
    output_dir = Path('soccer_visualizations')
    output_dir.mkdir(exist_ok=True)

    for i in range(10, len(data)):
        print(i)
        # Save lines visualization
        data.save_lines(i, output_dir)

        # Get and save entry visualizations
        entry = data[i]
        data.save_visualization(entry, i, output_dir)

        # If you want to save specific parts of the visualization:
        segs = entry['segments']
        save_to_disk(segs[:3]/2 + segs[3:]/2, output_dir /
                     f'combined_segments_{i}.jpg')
