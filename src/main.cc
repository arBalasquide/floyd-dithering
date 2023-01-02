/**
 * Program to convert images to look replicate the PC98 style
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sys/time.h>

using namespace cv;

// change this to match the bit RGB you want
// e.g 12-bit (4 bits per channel) is the PC98 spec and common in CRTs
#define COLOR_BITS 0

// Set COLOR_BITS to 0 to reduce colors by a fixed number through k-means
#define COLOR_CLUSTERS 2

// Do we want to actually dither or just reduce colors?
#define DITHER true

// Use Atkinson bias rather than Floyd-Steinberg
#define ATKINSON false

// Convert image to GRAYSCALE before reducing colors and dithering
#define GRAYSCALE false

Mat1f colors;

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

/**
 * Helper function to add error diffussion to each BGR channel
 */
Vec3b error_channel_diffussion(Vec3b pixel, const int32_t quant_error[],
                               double bias) {

  int32_t k[3] = {pixel[0], pixel[1], pixel[2]};

  // dither error diffusion on all 3 BGR channels
  for (int i = 0; i < 3; i++) {
    k[i] = clamp(pixel[i] + int32_t(float(quant_error[i] * bias)),0, 255);

    pixel[i] = k[i];
  }

  // the further rgb colors can be from each othere is around ~400
  float min_err = 65535;

  // we want to limit colors to our palette so find closest color
  for (int j = 0; j < COLOR_CLUSTERS; j++) {
    float dst = sqrt((colors(j, 0) - k[0]) * (colors(j, 0) - k[0]) +
                     (colors(j, 1) - k[1]) * (colors(j, 1) - k[1]) +
                     (colors(j, 2) - k[2]) * (colors(j, 2) - k[2]));

    if (dst < min_err) {
      min_err = dst;

      pixel[0] = uint8_t(colors(j, 0));
      pixel[1] = uint8_t(colors(j, 1));
      pixel[2] = uint8_t(colors(j, 2));
    }
  }

  return pixel;
}

/**
 * Add biased noise to the image to improve the quality of the reduced color
 * image
 */
void error_diffusion(Mat *img, const int32_t quant_error[], int x, int y) {

#if ATKINSON
  double a = 1.0f / 8.0f;
  double b = 1.0f / 8.0f;
  double c = 1.0f / 8.0f;
  double d = 1.0f / 8.0f;
#else
  double a = 7.0f / 16.0f;
  double b = 3.0f / 16.0f;
  double c = 5.0f / 16.0f;
  double d = 1.0f / 16.0f;
#endif

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

/**
 * K means to reduce color pallete to clusters rather than just limiting the
 * bits
 */
Mat reduce_color_pallete(Mat *img) {

  int k = COLOR_CLUSTERS;
  int n = img->rows * img->cols;

  Mat data = img->reshape(1, n);
  data.convertTo(data, CV_32F);

  std::vector<int> labels;
  kmeans(data, k, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, colors);

  for (int i = 0; i < n; i++) {

    data.at<float>(i, 0) = clamp(colors(labels[i], 0), 0, 255);
    data.at<float>(i, 1) = clamp(colors(labels[i], 1), 0, 255);
    data.at<float>(i, 2) = clamp(colors(labels[i], 2), 0, 255);
  }

  Mat reduced = data.reshape(3, img->rows);
  reduced.convertTo(reduced, CV_8U);

  return reduced;
}

Vec3b reduce_color_bits(Vec3b pixel) {

// I like how the second one changes the colors, the differences are a bit
// subtle try it out and see which one you like most
#if false
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
#else
  uchar maskBit = 0xFF;

  maskBit = maskBit << (8 - COLOR_BITS);

  uint8_t cb = pixel[0] & maskBit;
  uint8_t cg = pixel[1] & maskBit;
  uint8_t cr = pixel[2] & maskBit;

#endif

  Vec3b newpixel(cb, cg, cr);

  return newpixel;
}

// https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
void dither(Mat *img) {
  int rows = img->rows;
  int cols = img->cols;

#if !COLOR_BITS
  Mat reducedImg = reduce_color_pallete(img);
#endif

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      Vec3b pixel = img->at<Vec3b>(i, j);
#if !COLOR_BITS
      Vec3b newpixel = reducedImg.at<Vec3b>(i, j);
#else
      Vec3b newpixel = reduce_color_bits(pixel);
#endif

      // pixels are unsigned so we want to make sure we get negative values here
      // for error to get distributed properly
      const int32_t quant_error[3] = {int32_t(pixel[0] - newpixel[0]),
                                      int32_t(pixel[1] - newpixel[1]),
                                      int32_t(pixel[2] - newpixel[2])};

      // overwrite image with new pixel
      img->at<Vec3b>(i, j) = newpixel;

#if DITHER
      // since error diffussion goes back and forward, make sure we don't go out
      // of bound...
      if (i != 0 && j != cols) {
        error_diffusion(img, quant_error, i, j);
      }
#endif
    }
  }
}

int main(int argc, char **argv) {
  std::string filePath = "/home/adrian/ct.png";

  // TODO: Change to be an cli arg
  std::string image_path = samples::findFile(filePath);

#if GRAYSCALE
  Mat src = imread(image_path, IMREAD_GRAYSCALE);
#else
  Mat src = imread(image_path, IMREAD_COLOR);
#endif
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
