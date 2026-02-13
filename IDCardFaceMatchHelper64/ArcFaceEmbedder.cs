using IDCardFaceMatchHelper64.Detectors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.IO;
using System.Linq;
using Size = OpenCvSharp.Size;

namespace IDCardFaceMatchHelper64
{
    public class ArcFaceEmbedder : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly ScrfdFaceDetectorOrt _scrfd;

        private readonly Size _inputSize = new Size(112, 112);
        private readonly string _inputName;
        private readonly string _outputName;

        public ArcFaceEmbedder(
            string arcfaceOnnxPath,
            string scrfdOnnxPath,
            float faceConfThreshold = 0.5f,
            float faceNmsIou = 0.35f)
        {
            if (!File.Exists(arcfaceOnnxPath))
                throw new FileNotFoundException("ArcFace ONNX model not found.", arcfaceOnnxPath);

            if (!File.Exists(scrfdOnnxPath))
                throw new FileNotFoundException("SCRFD ONNX model not found.", scrfdOnnxPath);

            _session = new InferenceSession(arcfaceOnnxPath);
            _scrfd = new ScrfdFaceDetectorOrt(scrfdOnnxPath, faceConfThreshold, faceNmsIou);

            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();
        }

        public void Dispose()
        {
            _session?.Dispose();
            _scrfd?.Dispose();
        }

        public float[] GetEmbeddingFromFile(string imagePath)
        {
            using var img = Cv2.ImRead(imagePath);
            if (img.Empty())
                throw new InvalidOperationException("Unable to read image: " + imagePath);

            return GetEmbeddingFromMat(img);
        }

        public float[] GetEmbeddingFromMat(Mat bgr)
        {
            var det = _scrfd.DetectLargest(bgr);
            if (det is null)
                throw new Exception("No face detected.");

            var padded = PadRect(det.Value.Rect, bgr.Width, bgr.Height, 0.2);

            using var faceBgr = new Mat(bgr, padded).Clone();
            using var faceRgb = new Mat();
            Cv2.CvtColor(faceBgr, faceRgb, ColorConversionCodes.BGR2RGB);

            using var resized = new Mat();
            Cv2.Resize(faceRgb, resized, _inputSize);

            return RunArcFace(resized);
        }

        public (Rect rect, float[] embedding)[] GetAllFaceEmbeddings(string imagePath)
        {
            using var img = Cv2.ImRead(imagePath);
            if (img.Empty())
                throw new InvalidOperationException("Unable to read image: " + imagePath);

            return GetAllFaceEmbeddings(img);
        }

        public (Rect rect, float[] embedding)[] GetAllFaceEmbeddings(Mat bgr)
        {
            var dets = _scrfd.Detect(bgr);
            if (dets.Count == 0)
                return Array.Empty<(Rect, float[])>();

            // If you still get too many false positives:
            // - raise faceConfThreshold to 0.7+ when creating detector
            // - or uncomment this min-area filter:
            // dets = dets.Where(d => d.Rect.Width * d.Rect.Height > 40 * 40).ToList();

            return dets.Select(d =>
            {
                var padded = PadRect(d.Rect, bgr.Width, bgr.Height, 0.2);

                using var faceBgr = new Mat(bgr, padded).Clone();
                using var faceRgb = new Mat();
                Cv2.CvtColor(faceBgr, faceRgb, ColorConversionCodes.BGR2RGB);

                using var resized = new Mat();
                Cv2.Resize(faceRgb, resized, _inputSize);

                var emb = RunArcFace(resized);
                return (padded, emb);
            }).ToArray();
        }

        private static Rect PadRect(Rect r, int imgW, int imgH, double padFraction)
        {
            int padX = (int)(r.Width * padFraction);
            int padY = (int)(r.Height * padFraction);

            int x = Math.Max(0, r.X - padX);
            int y = Math.Max(0, r.Y - padY);
            int x2 = Math.Min(imgW, r.X + r.Width + padX);
            int y2 = Math.Min(imgH, r.Y + r.Height + padY);

            return new Rect(x, y, x2 - x, y2 - y);
        }

        /// <summary>
        /// ArcFace ONNX expects RGB 112x112, normalized, NHWC [1,112,112,3]
        /// </summary>
        private float[] RunArcFace(Mat faceRgb112)
        {
            var input = new DenseTensor<float>(new[] { 1, _inputSize.Height, _inputSize.Width, 3 });

            for (int y = 0; y < _inputSize.Height; y++)
                for (int x = 0; x < _inputSize.Width; x++)
                {
                    var pixel = faceRgb112.At<Vec3b>(y, x); // RGB
                    input[0, y, x, 0] = (pixel.Item0 - 127.5f) / 128.0f; // R
                    input[0, y, x, 1] = (pixel.Item1 - 127.5f) / 128.0f; // G
                    input[0, y, x, 2] = (pixel.Item2 - 127.5f) / 128.0f; // B
                }

            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_inputName, input) };
            using var results = _session.Run(inputs);

            var outputTensor = results.First(r => r.Name == _outputName)
                                      .AsEnumerable<float>()
                                      .ToArray();

            float norm = (float)Math.Sqrt(outputTensor.Sum(v => v * v));
            if (norm > 0)
            {
                for (int i = 0; i < outputTensor.Length; i++)
                    outputTensor[i] /= norm;
            }

            return outputTensor;
        }

        public static double CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Embedding sizes do not match.");

            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }

            if (na == 0 || nb == 0) return 0;
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
        }
    }
}