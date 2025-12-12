using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Linq;

namespace IDCardFaceMatchHelper64
{
    public sealed class ArcFaceEmbedder : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly CascadeClassifier _faceCascade;

        private static readonly Size ArcInputSize = new Size(112, 112);
        private readonly string _inputName;
        private readonly string _outputName;

        public ArcFaceEmbedder(string onnxModelPath, string haarCascadePath)
        {
            if (!File.Exists(onnxModelPath))
                throw new FileNotFoundException("ArcFace ONNX model not found.", onnxModelPath);

            if (!File.Exists(haarCascadePath))
                throw new FileNotFoundException("Haar cascade not found.", haarCascadePath);

            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            _session = new InferenceSession(onnxModelPath, opts);
            _faceCascade = new CascadeClassifier(haarCascadePath);


            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();

        }

        #region ---------- Public API ----------

        /// <summary>
        /// Extract largest face from image, return rect + BGR face crop.
        /// Caller must Dispose the returned Mat.
        /// </summary>
        public (Rect rect, Mat faceBgr) ExtractLargestFaceBgr(string imagePath, double pad = 0.15)
        {
            using var img = Cv2.ImRead(imagePath);
            if (img.Empty())
                throw new Exception("Unable to read image: " + imagePath);

            using var gray = new Mat();
            Cv2.CvtColor(img, gray, ColorConversionCodes.BGR2GRAY);
            Cv2.EqualizeHist(gray, gray);

            var faces = _faceCascade.DetectMultiScale(
                gray, 1.5, 4, 0, new Size(60, 60));

            if (faces.Length == 0)
                throw new Exception("No face detected in: " + imagePath);

            Rect best = faces[0];
            double bestArea = best.Width * best.Height;

            for (int i = 1; i < faces.Length; i++)
            {
                double a = faces[i].Width * faces[i].Height;
                if (a > bestArea)
                {
                    bestArea = a;
                    best = faces[i];
                }
            }

            int padX = (int)(best.Width * pad);
            int padY = (int)(best.Height * pad);

            int x = Math.Max(0, best.X - padX);
            int y = Math.Max(0, best.Y - padY);
            int x2 = Math.Min(img.Width, best.Right + padX);
            int y2 = Math.Min(img.Height, best.Bottom + padY);

            var padded = new Rect(x, y, x2 - x, y2 - y);
            var face = new Mat(img, padded).Clone(); // BGR

            return (padded, face);
        }

        /// <summary>
        /// Create ArcFace embedding from a BGR face crop.
        /// </summary>
        public float[] GetEmbeddingFromFaceMat(Mat faceBgr)
        {
            using var resized = new Mat();
            Cv2.Resize(faceBgr, resized, ArcInputSize);

            // ArcFace ONNX expects RGB
            using var rgb = new Mat();
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

            var emb = RunArcFace(rgb);
            L2NormalizeInPlace(emb);
            return emb;
        }

        /// <summary>
        /// Detect all faces in camera image and return rect + embedding for each.
        /// </summary>
        public (Rect rect, float[] embedding)[] GetAllFaceEmbeddings(string imagePath)
        {
            using var img = Cv2.ImRead(imagePath);
            if (img.Empty())
                return Array.Empty<(Rect, float[])>();

            using var gray = new Mat();
            Cv2.CvtColor(img, gray, ColorConversionCodes.BGR2GRAY);
            Cv2.EqualizeHist(gray, gray);

            var faces = _faceCascade.DetectMultiScale(
                gray, 1.5, 4, 0, new Size(60, 60));

            var list = new List<(Rect, float[])>();

            foreach (var r in faces)
            {
                using var face = new Mat(img, r);
                var emb = GetEmbeddingFromFaceMat(face);
                list.Add((r, emb));
            }

            return list.ToArray();
        }

        public static double CosineSimilarity(float[] a, float[] b)
        {
            double dot = 0;
            int n = Math.Min(a.Length, b.Length);
            for (int i = 0; i < n; i++)
                dot += a[i] * b[i];

            return dot; // vectors are L2-normalized
        }

        #endregion

        #region ---------- ArcFace Core ----------

        private float[] RunArcFace(Mat rgb112)
        {
            // NHWC: [1,112,112,3]
            var input = new DenseTensor<float>(new[] { 1, 112, 112, 3 });

            for (int y = 0; y < 112; y++)
            {
                for (int x = 0; x < 112; x++)
                {
                    var c = rgb112.At<Vec3b>(y, x); // RGB order

                    // normalize (keep your scheme; we can tune if needed)
                    input[0, y, x, 0] = (c.Item0 - 127.5f) / 128f; // R
                    input[0, y, x, 1] = (c.Item1 - 127.5f) / 128f; // G
                    input[0, y, x, 2] = (c.Item2 - 127.5f) / 128f; // B
                }
            }

            var inputs = new[]
            {
        NamedOnnxValue.CreateFromTensor(_inputName, input)
    };

            using var results = _session.Run(inputs, new[] { _outputName });
            return results.First().AsEnumerable<float>().ToArray();
        }





        public static void L2NormalizeInPlace(float[] v)
        {
            double sum = 0;
            for (int i = 0; i < v.Length; i++)
                sum += v[i] * v[i];

            double norm = Math.Sqrt(sum);
            if (norm < 1e-12) return;

            float inv = (float)(1.0 / norm);
            for (int i = 0; i < v.Length; i++)
                v[i] *= inv;
        }

        #endregion

        public void Dispose()
        {
            _session?.Dispose();
            _faceCascade?.Dispose();
        }
    }
}
