# image-interpolation

Deep learning project to create new images based on the previous ones

### Usage

#### Dataset

In order to be loaded using `NSRRDataLoader`, the dataset should be structured like so:

```
[root_dir]
│
└───View
│   │   img_1.png
│   │   img_2.png
│    ...
│   
└───Depth
│   │   img_1.png
│   │   img_2.png
│    ...
│   
└───Motion
│   │   img_1.png
│   │   img_2.png
│    ...
```

Where `root_dir` can be given as an argument, and `View`, `Depth` and `Motion` are static members of `NSRRDataLoader`.

**Note that corresponding tuples of (view, depth, motion) images files should share the same name, as they cannot be grouped together otherwise.**


#### Unit testing

    python3 unit_test.py --directory [path_to_root_dir] --filename [image_name]


### Miscellaneous information

Using :

* Pytorch project template at:
  https://github.com/victoresque/pytorch-template

* Pytorch implementation of SSIM:
  https://github.com/Po-Hsun-Su/pytorch-ssim

* Pytorch implementation of colour-space conversions:
  https://github.com/jorge-pessoa/pytorch-colors
