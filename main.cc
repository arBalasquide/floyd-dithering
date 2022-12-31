/**
 * Program to convert images to look replicate the PC98 style
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <bits/getopt_core.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sys/time.h>

using namespace cv;

// change this to match the bit RGB you want
// e.g 12-bit (4 bits per channel) is the PC98 spec and common in CRTs
#define COLOR_BITS 2

// Do we want to actually dither or just reduce colors?
#define DITHER true

/**
 * Do not let value exceed color range
 */
int clamp(float value, float min, float max) {

  if (value > max) {
    return max;
  } else if (value < min) {
    return min;
  }

  return value;
}

Vec3b error_channel_diffussion(Vec3b pixel, const int32_t quant_error[],
                               double bias) {

  int32_t k[3] = {pixel[0], pixel[1], pixel[2]};

  // dither error diffusion on all 3 BGR channels
  for (int i = 0; i < 3; i++) {
    k[i] += int32_t(float(quant_error[i] * bias));
  }

  pixel[0] = clamp(k[0], 0, 255);
  pixel[1] = clamp(k[1], 0, 255);
  pixel[2] = clamp(k[2], 0, 255);

  return pixel;
}

void error_diffusion(Mat *img, const int32_t quant_error[], int x, int y) {

  double a = 7.0f / 16.0f;
  double b = 3.0f / 16.0f;
  double c = 5.0f / 16.0f;
  double d = 1.0f / 16.0f;

  Vec3b pixel = img->at<Vec3b>(x + 1, y);
  img->at<Vec3b>(x + 1, y) = error_channel_diffussion(pixel, quant_error, a);

  pixel = img->at<Vec3b>(x - 1, y + 1);
  img->at<Vec3b>(x - 1, y + 1) =
      error_channel_diffussion(pixel, quant_error, b);

  pixel = img->at<Vec3b>(x, y + 1);
  img->at<Vec3b>(x, y + 1) = error_channel_diffussion(pixel, quant_error, c);

  pixel = img->at<Vec3b>(x + 1, y + 1);
  img->at<Vec3b>(x + 1, y + 1) =
      error_channel_diffussion(pixel, quant_error, d);
}

Vec3b find_closest_palette_color(Vec3b pixel) {

  const float fLevels = (1 << COLOR_BITS) - 1;

  // don't let values overflow, just clamp to max
  uint8_t cr = uint8_t(
      clamp(std::round(float(pixel[2]) / 255.0f * fLevels) / fLevels * 255.0f,
            0.0f, 255.0f));
  uint8_t cb = uint8_t(
      clamp(std::round(float(pixel[0]) / 255.0f * fLevels) / fLevels * 255.0f,
            0.0f, 255.0f));
  uint8_t cg = uint8_t(
      clamp(std::round(float(pixel[1]) / 255.0f * fLevels) / fLevels * 255.0f,
            0.0f, 255.0f));

  Vec3b newpixel(cb, cg, cr);

  return newpixel;
}

// https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
void dither(Mat *img) {
  int rows = img->rows - 1;
  int cols = img->cols - 1;

  for (int i = 1; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      Vec3b pixel = img->at<Vec3b>(i, j);
      Vec3b newpixel = find_closest_palette_color(pixel);

      // pixels are unsigned so we want to make sure we get negative values here
      // for error to get distributed properly
      const int32_t quant_error[3] = {(int32_t)pixel[0] - (int32_t)newpixel[0],
                                      (int32_t)pixel[1] - (int32_t)newpixel[1],
                                      (int32_t)pixel[2] - (int32_t)newpixel[2]};

      // overwrite image with new pixel
      img->at<Vec3b>(i, j) = newpixel;

#if DITHER
      error_diffusion(img, quant_error, i, j);
#endif
    }
  }
}

int main(int argc, char **argv) {
  char *filePath = "input.jpg";

  // TODO: Change to be an cli arg
  std::string image_path = samples::findFile(filePath);
  Mat src = imread(image_path, IMREAD_COLOR);

  if (src.empty()) {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }

  Mat img = src.clone();

  // TODO: Calc how much it took to generate the image
  std::cout << "Dithering image: " << image_path << std::endl;
  std::cout << "Bits per channel: " << COLOR_BITS << std::endl;
  std::cout << "Original img properties\nHeight: " << src.rows
            << "\tWidth: " << src.cols << std::endl;

  struct timeval current_time;
  gettimeofday(&current_time, NULL);

  dither(&img);

  struct timeval dither_time;
  gettimeofday(&dither_time, NULL);

  double time_diff = (dither_time.tv_sec + dither_time.tv_usec / 1E6) -
                     (current_time.tv_sec + current_time.tv_usec / 1E6);

  std::cout << "Time to process image: " << time_diff << "s" << std::endl;

  Mat img_copy = img.clone();
  Mat src_copy = src.clone();

  // TODO: This should keep the same aspect ratio of the original image
  resize(img, img_copy, Size(864, 480), 0, 0, INTER_LANCZOS4);
  resize(src, src_copy, Size(864, 480), 0, 0, INTER_LANCZOS4);

  Mat comp;
  hconcat(src_copy, img_copy, comp);

  // scale down to compare both, whilst writing the new one to the file system
  imshow("Comparison", comp);

  while ((waitKey() & 0xEFFFFF) != 27)
    ; // wait for ESC press to quit

  imwrite("output.png", img);

  return 0;
}
