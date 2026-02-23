using System;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace IDCardFaceMatchHelper64.Detectors
{
    /// <summary>
    /// ONNX anti-spoofing detector aligned with hairymax/Face-AntiSpoofing demo logic:
    /// - Frame BGR -> RGB
    /// - increased_crop: square crop with bbox_inc=1.5 and black padding
    /// - letterbox to 128 (or configured), CHW float32, /255
    /// - output: softmax -> score = prob(class0), label = argmax
    /// - decision: REAL if label==0 && score>threshold; UNKNOWN if label==0 && score<=threshold; else FAKE
    /// </summary>
    public sealed class AntiSpoofingDetectorOnnx : IDisposable
    {
        public enum SpoofLabel
        {
            Real = 0,
            Fake = 1,
            Unknown = 2
        }

        private readonly InferenceSession _session;
        private readonly string _inputName;

        private readonly int _inputSize;
        private readonly double _realThreshold;
        private readonly double _bboxInc;

        public AntiSpoofingDetectorOnnx(
            string onnxModelPath,
            int inputSize = 128,
            double realThreshold = 0.5,
            double bboxInc = 1.5)
        {
            if (!File.Exists(onnxModelPath))
                throw new FileNotFoundException("Anti-spoofing ONNX model not found.", onnxModelPath);

            _inputSize = inputSize;
            _realThreshold = realThreshold;
            _bboxInc = bboxInc;

            _session = new InferenceSession(onnxModelPath);
            _inputName = _session.InputMetadata.Keys.First();
        }

        public void Dispose() => _session?.Dispose();

        /// <summary>
        /// Evaluates using the repo's REAL/FAKE/UNKNOWN semantics.
        /// score is prob(class0) after softmax (same as pred[0][0] in python).
        /// </summary>
        public (SpoofLabel label, double scoreReal, double[] probs) Evaluate(Mat bgrFrame, Rect faceRect)
        {
            if (bgrFrame.Empty() || faceRect.Width <= 0 || faceRect.Height <= 0)
                return (SpoofLabel.Unknown, 0.0, Array.Empty<double>());

            // Python: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            using var rgbFrame = new Mat();
            Cv2.CvtColor(bgrFrame, rgbFrame, ColorConversionCodes.BGR2RGB);

            // Convert OpenCvSharp Rect(x,y,w,h) to python-like bbox tuple (x1,y1,x2,y2)
            int x1 = faceRect.X;
            int y1 = faceRect.Y;
            int x2 = faceRect.X + faceRect.Width;
            int y2 = faceRect.Y + faceRect.Height;

            // Python increased_crop(img, bbox, bbox_inc=1.5)
            using var cropRgb = IncreasedCrop(rgbFrame, x1, y1, x2, y2, _bboxInc);

            // AntiSpoof preprocessing: letterbox to inputSize, CHW, /255
            using var letterboxed = LetterboxToSquare(cropRgb, _inputSize);

            var inputTensor = MatRgbToCHWTensor(letterboxed);

            // Run ONNX
            using var results = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            });

            var raw = results.First().AsEnumerable<float>().ToArray();
            var probs = Softmax(raw);

            // Python:
            // pred = anti_spoof([...])[0]
            // score = pred[0][0]
            // label = np.argmax(pred)
            double scoreReal = probs.Length > 0 ? probs[0] : 0.0;
            int argmax = ArgMax(probs);

            SpoofLabel label;
            if (argmax == 0)
            {
                label = scoreReal > _realThreshold ? SpoofLabel.Real : SpoofLabel.Unknown;
            }
            else
            {
                label = SpoofLabel.Fake;
            }

            return (label, scoreReal, probs);
        }

        /// <summary>
        /// Implements python increased_crop exactly (conceptually):
        /// - bbox provided as x1,y1,x2,y2
        /// - make square with l=max(w,h), center at bbox center, expand by bboxInc
        /// - crop within image bounds and pad with black for out-of-bounds
        /// </summary>
        private static Mat IncreasedCrop(Mat rgbFrame, int x1, int y1, int x2, int y2, double bboxInc)
        {
            int realH = rgbFrame.Rows;
            int realW = rgbFrame.Cols;

            int w = x2 - x1;
            int h = y2 - y1;
            if (w <= 0 || h <= 0)
                return new Mat(); // empty

            int l = Math.Max(w, h);

            double xc = x1 + w / 2.0;
            double yc = y1 + h / 2.0;

            int x = (int)(xc - l * bboxInc / 2.0);
            int y = (int)(yc - l * bboxInc / 2.0);

            int cropSize = (int)(l * bboxInc);

            int x1c = x < 0 ? 0 : x;
            int y1c = y < 0 ? 0 : y;

            int x2c = (x + cropSize > realW) ? realW : (x + cropSize);
            int y2c = (y + cropSize > realH) ? realH : (y + cropSize);

            // Crop the valid region
            var roiRect = new Rect(x1c, y1c, Math.Max(0, x2c - x1c), Math.Max(0, y2c - y1c));
            using var cropped = roiRect.Width > 0 && roiRect.Height > 0
                ? new Mat(rgbFrame, roiRect).Clone()
                : new Mat(1, 1, MatType.CV_8UC3, Scalar.All(0));

            // Pad to cropSize x cropSize using black borders (cv2.copyMakeBorder)
            int top = y1c - y;                  // y1 - y
            int left = x1c - x;                 // x1 - x
            int bottom = cropSize - (y2c - y);  // int(l*bbox_inc - y2 + y)
            int right = cropSize - (x2c - x);   // int(l*bbox_inc - x2 + x)

            top = Math.Max(0, top);
            left = Math.Max(0, left);
            bottom = Math.Max(0, bottom);
            right = Math.Max(0, right);

            var padded = new Mat();
            Cv2.CopyMakeBorder(cropped, padded, top, bottom, left, right, BorderTypes.Constant, new Scalar(0, 0, 0));
            return padded;
        }

        /// <summary>
        /// Letterbox resize to newSize keeping aspect ratio, pad with black to newSize x newSize.
        /// This matches the repo preprocessing.
        /// </summary>
        private static Mat LetterboxToSquare(Mat rgb, int newSize)
        {
            if (rgb.Empty())
                return new Mat(newSize, newSize, MatType.CV_8UC3, Scalar.All(0));

            int oldH = rgb.Rows;
            int oldW = rgb.Cols;

            double ratio = (double)newSize / Math.Max(oldH, oldW);
            int scaledH = (int)Math.Round(oldH * ratio);
            int scaledW = (int)Math.Round(oldW * ratio);

            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new Size(scaledW, scaledH));

            int deltaW = newSize - scaledW;
            int deltaH = newSize - scaledH;

            int top = deltaH / 2;
            int bottom = deltaH - top;
            int left = deltaW / 2;
            int right = deltaW - left;

            var bordered = new Mat();
            Cv2.CopyMakeBorder(resized, bordered, top, bottom, left, right, BorderTypes.Constant, new Scalar(0, 0, 0));
            return bordered;
        }

        private DenseTensor<float> MatRgbToCHWTensor(Mat rgbSquare)
        {
            if (rgbSquare.Empty())
                return new DenseTensor<float>(new[] { 1, 3, _inputSize, _inputSize });

            if (rgbSquare.Rows != _inputSize || rgbSquare.Cols != _inputSize)
                throw new InvalidOperationException("Preprocess must return inputSize x inputSize.");

            var tensor = new DenseTensor<float>(new[] { 1, 3, _inputSize, _inputSize });

            for (int y = 0; y < _inputSize; y++)
            {
                for (int x = 0; x < _inputSize; x++)
                {
                    var px = rgbSquare.At<Vec3b>(y, x); // RGB
                    tensor[0, 0, y, x] = px.Item0 / 255f; // R
                    tensor[0, 1, y, x] = px.Item1 / 255f; // G
                    tensor[0, 2, y, x] = px.Item2 / 255f; // B
                }
            }

            return tensor;
        }

        private static double[] Softmax(float[] x)
        {
            if (x.Length == 0) return Array.Empty<double>();

            double max = x.Max();
            var exp = new double[x.Length];
            double sum = 0;

            for (int i = 0; i < x.Length; i++)
            {
                exp[i] = Math.Exp(x[i] - max);
                sum += exp[i];
            }

            if (sum <= 0) return exp.Select(_ => 0.0).ToArray();
            for (int i = 0; i < exp.Length; i++) exp[i] /= sum;
            return exp;
        }

        private static int ArgMax(double[] a)
        {
            if (a.Length == 0) return -1;
            int best = 0;
            for (int i = 1; i < a.Length; i++)
                if (a[i] > a[best]) best = i;
            return best;
        }
    }
}