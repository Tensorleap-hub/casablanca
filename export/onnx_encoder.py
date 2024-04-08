import torch
import torch.nn as nn
import numpy as np

from export.utils import Preprocessor
from networks.generator import Generator

checkpoint       = 'checkpoints/vox.pt'
size             = 256
latent_dim_style = 512
latent_dim_motion= 20
channel_multiplier=1
SRC_PATH = 'assets/images/58.jpg'
DRIVING_PATH = 'assets/videos/_8EjiNtSTCY_26651-27141-00000.mp4'


class LIAEncoder(nn.Module):
    def __init__(self, ckpt = 'checkpoints/vox.pt'):
        super(LIAEncoder, self).__init__()
        self.gen = Generator(size, latent_dim_style, latent_dim_motion, channel_multiplier).cuda()
        weight = torch.load(ckpt, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight,strict=False)
        self.gen.eval()


    def get_inputs(self, src_path, driving_path):
        preprocessor = Preprocessor(size)
        src = torch.from_numpy(preprocessor.image2np(src_path)).float().cuda()
        driving, fps = preprocessor.vid2np(driving_path)
        driving = torch.from_numpy(driving).float().cuda()
        return src, driving, fps

    @torch.no_grad()
    def forward(self, src, driving0, driving):
        h_start = self.gen.enc.enc_motion(driving0)
        wa, alpha, feats = self.gen.enc(src, driving, h_start)
        return wa, alpha, feats


def run_onnx(model_path, inputs):
    import onnx
    import onnxruntime as ort
    onnx_model = onnx.load(model_path)
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    input_names = [input.name for input in onnx_model.graph.input]

    output = ort_session.run(None, {'src':inputs[0].cpu().numpy(),
                                                          'driving_start':inputs[1].cpu().numpy(),
                                                          'driving':inputs[2].cpu().numpy()})
    return output


if __name__ == '__main__':
    lia = LIAEncoder()

    src = torch.rand(1,3,256,256).cuda()
    driving0 = torch.rand(1, 3, 256, 256).cuda()
    driving  = torch.rand(1, 3, 256, 256).cuda()

    torch.onnx.export(lia, (src,driving0,driving), "res/export_onnx/lia_encoder.onnx", input_names=['src','driving_start','driving'],
                      verbose=False)

    out_np = run_onnx("res/export_onnx/lia_encoder.onnx", (src, driving0, driving))
    out_torch = lia(src,driving0,driving)
    out_torch = [out_torch[0],*out_torch[1],*out_torch[2]]
    out_torch = [out.cpu().numpy() for out in out_torch]

    max_abs = np.max([np.max(np.abs(out_np[i] - out_torch[i])) for i in range(len(out_np))])
    print(f'Max Abs Error: {max_abs}')


