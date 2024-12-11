# Video Script

## Model

### HoVer-Net

- Proposed back in 2019

- Still used widely for nuclei segmentation tasks

- Chose HoverNet due to its interesting architecture, complex data processing and its well known nature.

- We chose not to fiddle with the architecture of the network, but rather just gain a understanding.

- Doubt there is much to improve.

### HoVer-NeXt

- HoVer-Net is 5 years old, the newest state-of-the-art HoVer-NeXt

- HoVer-NeXt is 17x faster than HoVer-Net but only 3% more accurate on the PanNuke dataset.

- U-Net architecture with two decoders.

## Performance

### Optimization Part 1

- We suspect it is due to either the possible gains being within the run-to-run variance.

- Or we need to optimize for significantly longer.

- We needed each optimization run to be 50 epochs since we expected to gain more in the end with a lowered learning rate + freezing.

- Unfortunately we could not get parallelization to work with raytune.

