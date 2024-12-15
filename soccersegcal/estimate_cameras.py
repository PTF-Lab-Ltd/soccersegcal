from soccersegcal.dataloader import SoccerNetFieldSegmentationDataset, save_seg_vis
from soccersegcal.pose import segs2cam
import torch
from sncalib.baseline_cameras import Camera
import numpy as np
from pathlib import Path
import json
from time import time
from scipy.optimize import fmin
from soccersegcal.train import LitSoccerFieldSegmentation
import fire
import os


def main(checkpoint_path="checkpoint.ckpt", indexes=None, data=None, out='cams_out', part='valid', show=False, overwrite=False):
    out_dir = Path(out)
    print(out_dir)
    (out_dir / part).mkdir(parents=True, exist_ok=True)

    os.mkdir(out_dir / 'vis_seg', exist_ok=True)

    world_scale = 100
    if data is None:
        data = list(SoccerNetFieldSegmentationDataset(width=960, split=part))
    if indexes is None:
        indexes = range(len(data))
    device = torch.device('cuda')
    print("Device:", device)
    print(torch.cuda.get_device_name())
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    segmentation_model = LitSoccerFieldSegmentation.load_from_checkpoint(
        checkpoint_path)
    segmentation_model.eval()
    segmentation_model = segmentation_model.to(device)  # Then move to GPU
    segmentation_model = segmentation_model.to(
        memory_format=torch.channels_last)  # Finally set memory format
    print("Model device:", next(segmentation_model.parameters()).device)
    for i in indexes:
        print("Index:", i)
        start_time = time()
        entry = data[i]
        ofn = out_dir / part / \
            ('camera_' + entry['name'].replace('.jpg', '.json'))
        pfn = out_dir / part / \
            f"camera_{int(entry['name'].split('.')[0]) - 1:05d}.json"
        if ofn.exists() and not overwrite:
            continue

        if pfn.exists():
            prev_cam = Camera()
            prev_cam.from_json_parameters(json.load(pfn.open()))
            prev_cam.position /= world_scale
            assert prev_cam.image_width == entry['image'].shape[-1]
        else:
            prev_cam = None

        if checkpoint_path is None:
            segs = entry['segments']
        else:
            img = entry['image'].to(device)
            print("Input tensor device:", img.device)
            print("Input tensor type:", img.dtype)
            with torch.no_grad():
                # unsqueeze is faster than [None]
                output = segmentation_model(img.unsqueeze(0))
                segs = torch.sigmoid_(output)[0].cpu()

            if show:
                image = entry['image']
                image_view = image
                segs_view = segs[:3] + segs[3:]
                # pview(image_view, pause=False)
                # pview(segs_view, pause=False)
                # pview(segs_view.to('cuda')-image_view, pause=False)
                # from vi3o.debugview import DebugViewer
                # out.view(imscale(np.hstack([a[0] for a in DebugViewer.named_viewers['Default'].image_array]), (720, 134)))
                save_seg_vis(entry['image'], out_dir / "vis_seg" / 'image.jpg')
                # segs = entry['segments']
                # save_to_disk(image_view, segs_view,  Path('cams_out') / entry['name'])
                # print(segs)

        # with torch.no_grad():
        ptz_model = segs2cam(segs, world_scale, prev_cam, show=show)
        if ptz_model is None:
            continue
        ptz_model = ptz_model.cpu()

        smalles_image_side = min(segs.shape[2], segs.shape[1])
        f = smalles_image_side / 2 / ptz_model.camera_focal.item()
        cam = Camera(segs.shape[2], segs.shape[1])
        cam.from_json_parameters({
            'position_meters': ptz_model.camera_position.detach().numpy() * world_scale,
            'principal_point': cam.principal_point,
            'x_focal_length': f,
            'y_focal_length': f,
            'pan_degrees': np.rad2deg(ptz_model.camera_pan.item()),
            'tilt_degrees': np.rad2deg(ptz_model.camera_tilt.item()),
            'roll_degrees': np.rad2deg(ptz_model.camera_roll.item()),
            'radial_distortion': ptz_model.radial_distortion.detach().numpy() if hasattr(ptz_model, 'radial_distortion') else np.zeros(6),
            'tangential_distortion': ptz_model.tangential_disto.detach().numpy() if hasattr(ptz_model, 'tangential_disto') else np.zeros(2),
            'thin_prism_distortion': ptz_model.thin_prism_disto.detach().numpy() if hasattr(ptz_model, 'thin_prism_disto') else np.zeros(4),
        })
        with open(ofn, "w") as fd:
            params = cam.to_json_parameters()
            if hasattr(ptz_model, 'mode_coeffs'):
                params['field_length'] = 105 + \
                    ptz_model.mode_coeffs[0].item() * 2 * world_scale
                params['field_width'] = 68 + \
                    ptz_model.mode_coeffs[1].item() * 2 * world_scale
            json.dump(params, fd)
        print("    ", time() - start_time, "s")


if __name__ == '__main__':
    fire.Fire(main)
