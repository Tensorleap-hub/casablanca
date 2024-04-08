import torch
import argparse
import os
from pathlib import Path
from export.utils import Preprocessor
from torchvision.io import write_video


class LIA:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.preprocessor = Preprocessor()

    @torch.no_grad()
    def animate(self, src_image_path, driving_video_path):
        src = self.preprocessor.image2torch(src_image_path)
        driving, fps = self.preprocessor.vid2torch(driving_video_path)

        frames = []
        for i in range(driving.shape[1]):
            driving0 = driving[:, 0]
            g_img = self.model(src, driving0, driving[:, i])
            frames.append(g_img)
        frames = [torch.cat([src, driving[:, i], frame], dim=-1).unsqueeze(1) for i, frame in enumerate(frames)]
        frames = torch.cat(frames, dim=1)

        # Normalize
        frames = ((frames * 0.5 + 0.5) * 255).clamp(0, 255).type('torch.ByteTensor')
        return frames[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default='assets/images/58.jpg')
    parser.add_argument("--driving_path", type=str, default='assets/videos/_8EjiNtSTCY_26651-27141-00000.mp4')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--out_dir", type=str, default='/Users/chenrothschild/Tensorleap/data/casablanca')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lia = LIA(args.model_path)

    frames = lia.animate(args.src_path, args.driving_path)
    write_video(str(Path(args.out_dir) / 'test.mp4'), frames.cpu().permute(0, 2, 3, 1), fps=30)
