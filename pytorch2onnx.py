import os
import torch
import onnx
import collections
import onnxruntime

from PIL import Image
from AWModel_single import AWModel
from onnxruntime.datasets import get_example

import torchvision.transforms as T


# replace InstanceNorm, AdaptiveAvgPooling2d and BatchNorm1d
def surgery(module):
    for n, c in module.named_children():
        if isinstance(c, torch.nn.InstanceNorm2d):
            c_ = InstanceNorm2d(c.weight.data, c.bias.data, c.eps)
            setattr(module, n, c_)
        elif isinstance(c, torch.nn.AdaptiveAvgPool2d):
            setattr(module, n, AdaptiveAvgPool2d())
        elif isinstance(c, torch.nn.BatchNorm1d) and 'bn_neck' in n:
            setattr(module, n, BatchNorm1d(c))


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def save_model(model, filepath):
    your_split = torch.split

    def my_split(x, y, dim=1):
        y = int(y)
        results = []
        for i in range(0, int(x.size(dim)), int(y)):
            results.append(eval(f'x[:,{i}:{i+y}]'))
        return tuple(results)
    torch.split = my_split   # walk around with the issue on tracing Tensor
    # make paths
    d = os.path.dirname(filepath)
    if d:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    x = torch.zeros(1, 3, 256, 128)
    model.eval().cpu()
    model.apply(surgery)

    with torch.no_grad():

        torch.onnx.export(model, x, 'test.onnx', export_params=True, output_names='o')
        converted_model = onnx.load('test.onnx')
        # onnx.checker.check_model(converted_model)

        # from wanghao, failed due to introduce wrong indexes for BN1d
        # for i, m in enumerate(model.model):
        #     # breakpoint()
        #     # m.pool_g = torch.nn.AvgPool2d([16, 8])
        #     print('%d params, %d flops' % compute_model_complexity(m, x.shape))
        #     torch.onnx.export(m, (x,), filepath + '_%d.onnx' % i, verbose=False, opset_version=11,
        #                       keep_initializers_as_inputs=True)


def test_onnx_pytorch(model_pytorch, model_onnx, img_path):
    test_transforms = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dummy_input = Image.open(img_path).convert('RGB')
    dummy_input = test_transforms(dummy_input)[None,]

    pytorch_out = model_pytorch(dummy_input)
    print('Pytorch output:')
    print(pytorch_out)

    # onnx 网络输出
    print('Onnx output')

    sess = onnxruntime.InferenceSession(model_onnx)
    onnx_out = sess.run(None, {'input.1': to_numpy(dummy_input)})
    print(onnx_out)

    return None


class InstanceNorm2d(torch.nn.Module):
    def __init__(self, weight, bias, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(weight[None, :, None])
        self.bias = torch.nn.Parameter(bias[None, :, None])
        self.eps = eps

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x_ = x.flatten(2)
        mean = x_.mean(2, keepdim=True)
        x_ = x_ - mean
        std = x_.pow(2).mean(2, keepdim=True).add(self.eps).sqrt()
        x_ = x_.div(std).mul(self.weight).add(self.bias)
        return x_.reshape(shape)


class AdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(3, keepdim=True).mean(2, keepdim=True)


class BatchNorm1d(torch.nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.weight = torch.nn.Parameter(bn.weight.data[None, :])
        self.bias = torch.nn.Parameter(bn.bias.data[None, :])
        self.eps = bn.eps
        self.running_mean = torch.nn.Parameter(bn.running_mean.data[None, :])
        self.running_var = torch.nn.Parameter(bn.running_var.data[None, :])

    def forward(self, x):
        std = self.running_var.add(self.eps).sqrt()
        return x.sub(self.running_mean).div(std).mul(self.weight).add(self.bias)


if __name__ == '__main__':

    # load corresponding model
    model = AWModel()
    state_dict = torch.load('../ours/test2.pt')

    load_metadata = collections.OrderedDict()
    for name, child in state_dict.items():
        if 'classifier' not in name:
            # name = name[0:5] + name[7:]
            name = name[7:]
            load_metadata[name] = child

    model.load_state_dict(load_metadata, strict=False)
    model.eval()

    # model conversion
    save_model(model, 'test' + '.pt')

    example_model = get_example('/home/wzk/Desktop/ReID_Nantong/test_wanghao/model_convert/test.onnx')
    test_onnx_pytorch(model_pytorch=model, model_onnx=example_model,
                      img_path='/home/wzk/Datasets/DukeMTMC-reID/query/0005_c2_f0046985.jpg')
