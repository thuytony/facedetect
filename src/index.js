/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import '@tensorflow-models/face-detection';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE, createDetector} from './shared/params';
import {setupStats} from './shared/stats_panel';
import {setBackendAndEnvFlags} from './shared/util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
let eyebrowImage;

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimateFaceStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimateFaceStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let faces = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateFaces.
    beginEstimateFaceStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      faces =
          await detector.estimateFaces(camera.video, {flipHorizontal: false});
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimateFaceStats();
  }

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.

  // if (faces && faces.length > 0 && !STATE.isModelChanged) {
  //   camera.drawResults(
  //       faces, STATE.modelConfig.triangulateMesh,
  //       STATE.modelConfig.boundingBox);
  // }

  // if (faces && faces.length > 0 && !STATE.isModelChanged) {
  //   camera.drawResults(
  //       faces, STATE.modelConfig.triangulateMesh,
  //       STATE.modelConfig.boundingBox);

  //   for (const face of faces) {
  //     const keypoints = face.keypoints;
  //     const leftEyebrow = keypoints.find(point => point.name === 'leftEyebrow');
  //     const rightEyebrow = keypoints.find(point => point.name === 'rightEyebrow');

  //     if (leftEyebrow && rightEyebrow) {
  //       drawEyebrow(camera.ctx, leftEyebrow, 'left');
  //       drawEyebrow(camera.ctx, rightEyebrow, 'right');
  //     }
  //   }
  // }

  if (faces && faces.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(
        faces, STATE.modelConfig.triangulateMesh,
        STATE.modelConfig.boundingBox);

    // for (const face of faces) {
    //   const keypoints = face.keypoints;
    //   const leftEyebrow = keypoints.find(point => point.name === 'leftEyebrow');
    //   const rightEyebrow = keypoints.find(point => point.name === 'rightEyebrow');
    //   const leftEye = keypoints.find(point => point.name === 'leftEye');
    //   const rightEye = keypoints.find(point => point.name === 'rightEye');

    //   if (leftEyebrow && rightEyebrow && leftEye && rightEye) {
    //     drawEyebrow(camera.ctx, leftEyebrow, rightEyebrow, leftEye, rightEye, 'left');
    //     drawEyebrow(camera.ctx, leftEyebrow, rightEyebrow, leftEye, rightEye, 'right');
    //   }
    // }
  }
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
};

function loadEyebrowImage() {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = 'https://i.ibb.co/4Kt2msK/brows.png'; // Replace with the path to your eyebrow image
    img.onload = () => resolve(img);
    img.onerror = (error) => reject(error);
  });
}

// function drawEyebrow(ctx, eyebrowPoint, side) {
//   if (eyebrowImage) {
//     const eyebrowWidth = 40; // Adjust as necessary
//     const eyebrowHeight = eyebrowWidth / eyebrowImage.width * eyebrowImage.height;
//     let x, y;

//     if (side === 'left') {
//       x = eyebrowPoint.x - eyebrowWidth / 2 - 13;
//       y = eyebrowPoint.y - eyebrowHeight / 2;
//     } else {
//       x = eyebrowPoint.x - eyebrowWidth / 2 + 13;
//       y = eyebrowPoint.y - eyebrowHeight / 2;
//     }

//     ctx.drawImage(eyebrowImage, x, y, eyebrowWidth, eyebrowHeight);
//   }
// }

function drawEyebrow(ctx, leftEyebrow, rightEyebrow, leftEye, rightEye, side) {
  if (eyebrowImage) {
    // Calculate angle of rotation based on the eyes position
    const dx = rightEye.x - leftEye.x;
    const dy = rightEye.y - leftEye.y;
    const angle = Math.atan2(dy, dx);

    // Calculate distance between eyes and use it to scale eyebrow
    const eyeDistance = Math.sqrt(dx * dx + dy * dy);
    const scale = eyeDistance / 100; // Adjust this value based on your image dimensions

    // Determine the position and size of the eyebrow image
    let x, y, centerX, centerY;
    if (side === 'left') {
      centerX = leftEyebrow.x;
      centerY = leftEyebrow.y;
    } else {
      centerX = rightEyebrow.x;
      centerY = rightEyebrow.y;
    }
    
    const eyebrowWidth = Math.abs(dx) * 0.5; // Adjust as necessary
    const eyebrowHeight = eyebrowWidth / eyebrowImage.width * eyebrowImage.height;

    // Determine the position and size of the eyebrow
    let eyebrowStart, eyebrowEnd;
    if (side === 'left') {
      eyebrowStart = leftEyebrow;
      eyebrowEnd = {x: leftEyebrow.x + eyeDistance * 0.3, y: leftEyebrow.y};
    } else {
      eyebrowStart = rightEyebrow;
      eyebrowEnd = {x: rightEyebrow.x - eyeDistance * 0.3, y: rightEyebrow.y};
    }

    // Save the current state of the canvas
    ctx.save();

    // Translate to the center of the eyebrow
    ctx.translate(centerX, centerY);

    // Rotate the canvas around the center of the eyebrow
    // ctx.rotate(angle);

    // // Draw the image centered at the translated and rotated origin
    ctx.drawImage(eyebrowImage, -eyebrowWidth / 2, -eyebrowHeight / 2, eyebrowWidth, eyebrowHeight);

    // Restore the canvas state
    ctx.restore();
  }
}

async function app() {
  console.log('----------- app start -------------')
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);

  await setupDatGui(urlParams);

  stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  eyebrowImage = await loadEyebrowImage();

  renderPrediction();
};

app();
