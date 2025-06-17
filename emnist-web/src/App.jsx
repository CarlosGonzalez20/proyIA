import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import cv from 'opencv-ts';
import './App.css';

const EMNIST_MAPPING = [
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
  'U', 'V', 'W', 'X', 'Y', 'Z',
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
  'u', 'v', 'w', 'x', 'y', 'z'
];

function App() {
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsModelLoading(true);
        setError(null);
        console.log("Esperando que OpenCV cargue...");

        await new Promise((resolve) => {
          if (cv.getBuildInformation) {
            resolve();
          } else {
            cv['onRuntimeInitialized'] = () => {
              resolve();
            };
          }
        });

        class L2 extends tf.serialization.Serializable {
          constructor(config) {
            super();
            this.l2 = config.l2 != null ? config.l2 : 0.01;
          }
          apply(x) {
            return tf.mul(this.l2, tf.sum(tf.square(x)));
          }
          getConfig() {
            return { l2: this.l2 };
          }
          static className = 'L2';
        }
        tf.serialization.registerClass(L2);

        const loadedModel = await tf.loadLayersModel('/modelo_json/model.json');
        setModel(loadedModel);
        console.log("Modelo cargado correctamente");
      } catch (err) {
        console.error("Error cargando OpenCV o el modelo:", err);
        setError(`Error al cargar dependencias: ${err.message}`);
      } finally {
        setIsModelLoading(false);
      }
    };

    loadModel();
  }, []);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let drawing = false;
    let lastX = 0;
    let lastY = 0;
    let hasDrawn = false;
    let timeoutId = null;

    const setupCanvas = () => {
      const container = canvas.parentElement;
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 15;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
    };

    const startDrawing = (e) => {
      drawing = true;
      [lastX, lastY] = [e.offsetX, e.offsetY];
      hasDrawn = false;
      if (timeoutId) clearTimeout(timeoutId);
    };

    const draw = (e) => {
      if (!drawing) return;
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(e.offsetX, e.offsetY);
      ctx.stroke();
      [lastX, lastY] = [e.offsetX, e.offsetY];
      hasDrawn = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    };

    const stopDrawing = () => {
      drawing = false;
      if (!hasDrawn) return;
      if (timeoutId) clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        predictAll();
        hasDrawn = false;
      }, 3000);
    };

    setupCanvas();

    window.addEventListener('resize', setupCanvas);
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    return () => {
      window.removeEventListener('resize', setupCanvas);
      canvas.removeEventListener('mousedown', startDrawing);
      canvas.removeEventListener('mousemove', draw);
      canvas.removeEventListener('mouseup', stopDrawing);
      canvas.removeEventListener('mouseout', stopDrawing);
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [model]);

  const predictAll = async () => {
    if (!model || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    try {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(canvas, 0, 0);

      const src = cv.imread(tempCanvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const binary = new cv.Mat();
      cv.threshold(gray, binary, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(binary.clone(), contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      let boundingRects = [];

      for (let i = 0; i < contours.size(); ++i) {
        const contour = contours.get(i);
        const rect = cv.boundingRect(contour);
        if (rect.width >= 20 && rect.height >= 20) {
          boundingRects.push({ contour, rect });
        }
      }

      boundingRects.sort((a, b) => a.rect.x - b.rect.x);

      for (let i = 0; i < boundingRects.length; ++i) {
        const { contour, rect } = boundingRects[i];
        const roi = binary.roi(rect);
        const resized = new cv.Mat();
        cv.resize(roi, resized, new cv.Size(28, 28), 0, 0, cv.INTER_AREA);

        const inputTensor = tf.tidy(() => {
          const imageData = new ImageData(28, 28);
          for (let j = 0; j < 28 * 28; j++) {
            const val = resized.data[j];
            imageData.data[j * 4] = val;
            imageData.data[j * 4 + 1] = val;
            imageData.data[j * 4 + 2] = val;
            imageData.data[j * 4 + 3] = 255;
          }

          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = 28;
          tempCanvas.height = 28;
          const tempCtx = tempCanvas.getContext('2d');
          tempCtx.putImageData(imageData, 0, 0);

          return tf.browser.fromPixels(tempCanvas, 1)
            .toFloat()
            .div(255.0)
            .reshape([1, 28, 28, 1]);
        });

        const pred = model.predict(inputTensor);
        const idx = pred.argMax(-1).dataSync()[0];
        const char = EMNIST_MAPPING[idx] || '?';

        ctx.fillStyle = 'white';
        ctx.fillRect(rect.x, rect.y, rect.width, rect.height);

        ctx.fillStyle = 'blue';
        ctx.font = `${Math.min(rect.width, rect.height)}px Arial`;
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'center';
        ctx.fillText(char, rect.x + rect.width / 2, rect.y + rect.height / 2);

        roi.delete();
        resized.delete();
        inputTensor.dispose();
        pred.dispose();
      }

      src.delete();
      gray.delete();
      binary.delete();
      contours.delete();
      hierarchy.delete();
    } catch (err) {
      console.error("Error en la predicción:", err);
      setError(`Error en la predicción: ${err.message}`);
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  return (
    <div className="app-container">
      <h1>Pizarra Inteligente EMNIST</h1>
      {error && <div className="error-message">{error}</div>}
      <div className="canvas-container">
        <canvas ref={canvasRef} className="drawing-canvas" />
      </div>
      <div className="controls">
        <button onClick={clearCanvas} disabled={isModelLoading}>
          {isModelLoading ? "Cargando modelo..." : "Limpiar Pizarra"}
        </button>
        {isModelLoading && <div className="loading">Cargando modelo, por favor espere...</div>}
      </div>
    </div>
  );
}

export default App;
