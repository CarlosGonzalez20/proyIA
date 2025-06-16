import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import cv from 'opencv-ts';
import './App.css';

// Registrar la clase del regularizador L2
// class L2 extends tf.regularizers.L1L2 {
//     static className = 'L2';
// }
// tf.serialization.registerClass(L2);

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
  const [timerID, setTimerID] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [error, setError] = useState(null);

  // Cargar el modelo
  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsModelLoading(true);
        setError(null);
        console.log("Cargando el modelo...");
        
        // Verificar que OpenCV esté cargado
        if (!cv.getBuildInformation) {
          throw new Error("OpenCV no se ha cargado correctamente");
        }

        // Cargar el modelo
        const loadedModel = await tf.loadLayersModel('/modelo_json/model.json');
        setModel(loadedModel);
        console.log("Modelo cargado correctamente");
      } catch (err) {
        console.error("Error cargando el modelo:", err);
        setError(`Error al cargar el modelo: ${err.message}`);
      } finally {
        setIsModelLoading(false);
      }
    };

    loadModel();

    return () => {
      if (model) {
        model.dispose();
      }
    };
  }, []);

  // Configurar canvas y eventos de dibujo
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let drawing = false;
    let lastX = 0;
    let lastY = 0;

    const setupCanvas = () => {
      const container = canvas.parentElement;
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      
      // Rellenar el canvas de blanco
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Configuración inicial del dibujo
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 15;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
    };

    const startDrawing = (e) => {
      drawing = true;
      [lastX, lastY] = [e.offsetX, e.offsetY];
    };

    const draw = (e) => {
      if (!drawing) return;
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(e.offsetX, e.offsetY);
      ctx.stroke();
      [lastX, lastY] = [e.offsetX, e.offsetY];
    };

    const stopDrawing = () => {
      drawing = false;
      clearTimeout(timerID);
      setTimerID(setTimeout(predictAll, 2000));
    };

    // Configuración inicial
    setupCanvas();

    // Event listeners
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
      clearTimeout(timerID);
    };
  }, [model]);

  const predictAll = async () => {
    if (!model || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    try {
      // Crear una copia del canvas para procesar
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(canvas, 0, 0);

      // Procesamiento con OpenCV
      const src = cv.imread(tempCanvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const binary = new cv.Mat();
      cv.threshold(gray, binary, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

      // Encontrar contornos
      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(binary.clone(), contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      // Procesar cada contorno
      for (let i = 0; i < contours.size(); ++i) {
        const contour = contours.get(i);
        const rect = cv.boundingRect(contour);

        // Filtrar contornos pequeños
        if (rect.width < 20 || rect.height < 20) continue;

        // Extraer ROI
        const roi = binary.roi(rect);
        const resized = new cv.Mat();
        cv.resize(roi, resized, new cv.Size(28, 28), 0, 0, cv.INTER_AREA);

        // Convertir a tensor
        const inputTensor = tf.tidy(() => {
          const imageData = new ImageData(28, 28);
          for (let i = 0; i < 28 * 28; i++) {
            const val = resized.data[i];
            imageData.data[i * 4] = val;
            imageData.data[i * 4 + 1] = val;
            imageData.data[i * 4 + 2] = val;
            imageData.data[i * 4 + 3] = 255;
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

        // Predecir
        const pred = model.predict(inputTensor);
        const idx = pred.argMax(-1).dataSync()[0];
        const char = EMNIST_MAPPING[idx] || '?';

        // Dibujar el resultado
        ctx.clearRect(rect.x, rect.y, rect.width, rect.height);
        ctx.fillStyle = 'blue';
        ctx.font = `${Math.min(rect.width, rect.height)}px Arial`;
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'center';
        ctx.fillText(char, rect.x + rect.width / 2, rect.y + rect.height / 2);

        // Liberar memoria
        roi.delete();
        resized.delete();
        inputTensor.dispose();
        pred.dispose();
      }

      // Liberar memoria
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
        <canvas 
          ref={canvasRef} 
          className="drawing-canvas"
        />
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