from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from networks.generator import Generator

checkpoint       = 'checkpoints/vox.pt'
size             = 256
latent_dim_style = 512
latent_dim_motion= 20
channel_multiplier=1

DecoderInput = namedtuple('DecoderInput', ['wa', 'alpha0', 'alpha1', 'alpha2', 'feats8',
                                               'feats16', 'feats32', 'feats64', 'feats128', 'feats256'])

class LIADecoder(nn.Module):
    def __init__(self, ckpt = 'checkpoints/vox.pt'):
        super(LIADecoder, self).__init__()
        self.gen = Generator(size, latent_dim_style, latent_dim_motion, channel_multiplier).cuda()
        weight = torch.load(ckpt, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        # Perform QR decomp for decoder direction
        dir = self.gen.dec.direction
        Q, R = torch.linalg.qr(dir.weight + 1e-8)
        self.Q = nn.Parameter(Q).cuda()
        dir.forward = self._direction_patched

    def _direction_patched(self, input):
        if input is None:
            return self.Q
        else:
            out = torch.einsum('bi,ij->bij',input,self.Q.T)
            out = torch.sum(out, dim=1)

            return out

    @classmethod
    def get_random_inputs(cls):
        wa = torch.randn(1,latent_dim_style).cuda()
        alpha = [torch.randn(1,latent_dim_motion).cuda() for _ in range(3)]
        feats_sizes = {'8': 512, '16': 512, '32': 512, '64': 256, '128': 128, '256': 64}
        feats = []
        for size, channels in feats_sizes.items():
            feats.append(torch.randn(1,channels,int(size),int(size)).cuda())

        return DecoderInput(wa, *alpha, *feats)

    @torch.no_grad()
    def forward(self, *inps):
        wa = inps[0]
        alpha = inps[1:4]
        feats = inps[4:]

        img_recon = self.gen.dec(wa, alpha, feats)
        return img_recon

def run_onnx(model_path, inputs):
    import onnx
    import onnxruntime as ort
    onnx_model = onnx.load(model_path)
    ort_session = ort.InferenceSession(model_path,providers=['CUDAExecutionProvider'])

    # Generate some dummy input data
    inputs = DecoderInput(*[inp.cpu().numpy() for inp in inputs])
    output = ort_session.run(None, inputs._asdict())[0]
    return output

if __name__ == '__main__':
    lia = LIADecoder()

    decoder_input: DecoderInput = LIADecoder.get_random_inputs()

    torch.onnx.export(lia, (decoder_input), "res/export_onnx/lia_decoder.onnx", input_names=DecoderInput._fields,
                      verbose=False)

    decoder_input = LIADecoder.get_random_inputs()
    np_out = run_onnx("res/export_onnx/lia_decoder.onnx", decoder_input)
    torch_out = lia(*decoder_input).cpu().numpy()

    print(f' Max error in output: {np.max(np.abs(np_out - torch_out)):.4f}')
