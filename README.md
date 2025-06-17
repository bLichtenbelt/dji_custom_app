# DJI Custom App

This repository contains a customized version of DJI's Mobile SDK sample application for Android.  The sample code under `SampleCode-V5` demonstrates how to load and run an ONNX model on frames from the drone's video stream and optionally blur detected objects.

## Key Features

- **ONNX model loader** – `OnnxModelLoader.kt` loads a model from app assets or from a file on external storage.
- **Model runner** – `ModelRunner.kt` converts YUV frames, performs inference with the loaded model and applies gaussian blur.  Blur labels are read from `blur_config.json`.
- **Video stream processing** – `VideoChannelFragment` calls `ModelRunner.runAndBlur` for each received frame so detections are blurred live.
- **Configurable blur** – `blur_config.json` defines which labels should be blurred.  Example:
  ```json
  {
    "blur_labels": ["person", "car"]
  }
  ```
- **Gradle setup** – The project targets Android SDK 34 and uses Kotlin 1.8.10.  Onnxruntime 1.22.0 is included as a dependency.

## Directory Layout

- `Documentation` – additional documentation files.
- `SampleCode-V5/android-sdk-v5-sample` – Android application source code.
- `SampleCode-V5/android-sdk-v5-as` – build configuration for the sample.

## Building

Open the project in Android Studio using the files in `SampleCode-V5/android-sdk-v5-as`.  Gradle properties specify `ANDROID_COMPILE_SDK_VERSION=34` and `NDK_VERSION=21.4.7075529`.  Provide your DJI and map API keys if required, then sync and build.

## Usage

1. Place an ONNX model in the app `assets` directory or on the device's external storage.
2. Adjust `blur_config.json` to list the detection labels to blur.
3. Build and run the app on a supported device.
4. Incoming video frames will be processed; regions corresponding to configured labels are blurred before display.

See `LICENSE.txt` for licensing details.
