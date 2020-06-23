# DIFRINT [TOG20 / SIGGRAPH Asia19]
This is the test code reference implementation of Deep Iterative Frame Interpolation for Full-frame Video Stabilization [1], using PyTorch.
This work proposes a full-frame video stabilization method via frame interpolation techniques, making use of a self-supervised deep learning approach.
Should you make use of our work, please cite our paper [1].

Our paper can be found in the <a href="https://dl.acm.org/doi/abs/10.1145/3363550">ACM Digital Library</a> and <a href="https://arxiv.org/abs/1909.02641">arXiv</a>.

## Setup
We used the following python and package versions:

`python==3.5.6`
`torch==1.0.0`
`cupy==4.1.0`
`pillow==5.2.0`
`numpy==1.15.2`
`matplotlib==3.0.0`
`pypng==0.0.20`
`opencv-contrib-python==4.1.0.25`
`CUDA==9.0`

You may require to setup the correlation package for computing the cost volume module in PWC-Net.
If required, please follow the instructions in <a href="https://github.com/vt-vl-lab/pwc-net.pytorch">vt-vl-lab/pwc-net</a>.

## Usage
You can run `python run_seq2.py --cuda --n_iter 3 --skip 2` to obtain example results on a sample given in the `data` folder, which will be saved in the `output` folder.

By default, our experiments were done with 3 iterations and skip parameter of 2.
This can be customized by adjusting the `--n_iter` and `--skip` options.

We also provide code for making .avi videos from output frames, and a reference code for quality metrics.

## Supplementary video
Please refer to the supplementary video provided below (click thumbnail):

<a href="https://youtu.be/qXi9NXOvIgM" rel="Video"><img src="http://img.youtube.com/vi/qXi9NXOvIgM/0.jpg" alt="Video" width="50%"></a>


## References
```
[1] @article{Choi_TOG20,
	author = {Choi, Jinsoo and Kweon, In So},
	title = {Deep Iterative Frame Interpolation for Full-Frame Video Stabilization},
	year = {2020},
	issue_date = {February 2020},
	publisher = {Association for Computing Machinery},
	volume = {39},
	number = {1},
	issn = {0730-0301},
	url = {https://doi.org/10.1145/3363550},
	journal = {ACM Transactions on Graphics},
	articleno = {4},
	numpages = {9},
    }
```

## License
The provided implementation is strictly for academic purposes only. 
Should you be interested in using our technology for any commercial use, please contact us.