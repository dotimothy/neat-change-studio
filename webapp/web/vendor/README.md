# Vendored files

| file | what | source | license |
|---|---|---|---|
| `opencv-4.12.0.js` | OpenCV.js 4.12.0 (OpenCV compiled to WebAssembly), used by `../align-worker.js` for the webcam's alignment check | npm `@techstark/opencv-js@4.12.0-release.1`, `dist/opencv.js`, unchanged | Apache-2.0 (OpenCV, https://opencv.org) |

SHA-256 of `opencv-4.12.0.js`: `bd0c3e6448043de04f6a64a12cb7b759f78c3ab8f7c35c9f2e0f71c88bb17103`

It is loaded only when the webcam panel needs it, inside a Web Worker, and is served by the
board, so the demo needs no internet access.
