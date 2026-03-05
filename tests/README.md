# Test Data Conventions

Integration tests look for reference artifacts at these default paths:

- `reference/pytorch-i3d/models/rgb_imagenet.pt`
- `reference/kinetics_i3d_pytorch/model/model_rgb.pth`
- `reference/kinetics_i3d_pytorch/data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy`
- `reference/kinetics_i3d_pytorch/data/kinetic-samples/label_map.txt`

You can override each path with environment variables:

- `I3D_TEST_RGB_PYTORCH_CKPT`
- `I3D_TEST_RGB_KINETICS_CKPT`
- `I3D_TEST_RGB_SAMPLE_NPY`
- `I3D_TEST_LABEL_MAP`
