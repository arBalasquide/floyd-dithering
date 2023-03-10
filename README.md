# floyd-dithering

Simple C++ program to reduce colors of an image and applying Floyd-Steinberg Dithering algorithm.

## Build

```
mkdir build
cd build
cmake ..
make
```

## Usage

Right now, all the inputs and options are hardcoded into the program. Change `filePath` to the path of the image you want to dither.

You may also change `COLOR_BITS` define to change how many bits there are per color. The less, the more washed up the image will be. If you want to use k-means to reduce colors and have a more specific amount of colors, set `COLOR_BITS` to 0 and change `COLOR_CLUSTERS` to the limit of colors you want.

You can also disable dithering through the `DITHER` define.

You can run the program simply by running:

`./dither`

Press `ESC` to leave the window and save the file.

An image will be created in your current directory called `output.png`.
