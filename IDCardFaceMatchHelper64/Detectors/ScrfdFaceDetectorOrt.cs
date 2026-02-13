using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace IDCardFaceMatchHelper64.Detectors
{
    public sealed class ScrfdFaceDetectorOrt : IDisposable
    {
        public readonly record struct Detection(Rect Rect, float Confidence);

        private readonly InferenceSession _session;
        private readonly string _inputName;

        private readonly int _inH;
        private readonly int _inW;

        private readonly float _confThreshold;
        private readonly float _nmsIouThreshold;

        private readonly int _numAnchors; // SCRFD 2-anchor variant outputs 9 heads

        private static readonly int[] Strides = { 8, 16, 32 };

        public ScrfdFaceDetectorOrt(string scrfdOnnxPath, float confThreshold = 0.5f, float nmsIouThreshold = 0.35f)
        {
            if (!File.Exists(scrfdOnnxPath))
                throw new FileNotFoundException("SCRFD model not found.", scrfdOnnxPath);

            _session = new InferenceSession(scrfdOnnxPath);
            _inputName = _session.InputMetadata.Keys.First();

            // SCRFD is NCHW in most exports: [1,3,H,W]
            var dims = _session.InputMetadata[_inputName].Dimensions;
            if (dims.Length >= 4 && dims[2] > 0 && dims[3] > 0)
            {
                _inH = (int)dims[2];
                _inW = (int)dims[3];
            }
            else
            {
                _inH = 640;
                _inW = 640;
            }

            _confThreshold = confThreshold;
            _nmsIouThreshold = nmsIouThreshold;

            // Expect SCRFD named heads:
            // score_8/16/32, bbox_8/16/32, kps_8/16/32
            var outs = _session.OutputMetadata.Keys.ToArray();
            bool hasNamed = outs.Contains("score_8") && outs.Contains("bbox_8") && outs.Contains("kps_8")
                         && outs.Contains("score_16") && outs.Contains("bbox_16") && outs.Contains("kps_16")
                         && outs.Contains("score_32") && outs.Contains("bbox_32") && outs.Contains("kps_32");

            if (!hasNamed)
                throw new InvalidOperationException("SCRFD outputs not found as score_*/bbox_*/kps_*.");

            _numAnchors = outs.Length == 9 ? 2 : 1;
        }

        public void Dispose() => _session?.Dispose();

        public List<Detection> Detect(Mat frameBgr)
        {
            if (frameBgr == null || frameBgr.Empty())
                return new List<Detection>();

            int srcH = frameBgr.Rows;
            int srcW = frameBgr.Cols;

            PrepareScrfdInput(frameBgr, out DenseTensor<float> input, out float detScale);

            var dets = new List<Detection>();

            // DO NOT dispose NamedOnnxValue in your ORT version
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_inputName, input) };

            using var results = _session.Run(inputs);

            var outDict = results.ToDictionary(
                r => r.Name,
                r => r.AsEnumerable<float>().ToArray(),
                StringComparer.OrdinalIgnoreCase);

            foreach (var stride in Strides)
            {
                if (!outDict.TryGetValue($"score_{stride}", out var scoresRaw)) continue;
                if (!outDict.TryGetValue($"bbox_{stride}", out var bboxesRaw)) continue;

                int n = Math.Min(scoresRaw.Length, bboxesRaw.Length / 4);
                if (n <= 0) continue;

                float max = scoresRaw.Take(n).Max();
                float min = scoresRaw.Take(n).Min();
                bool usingSigmoid = max > 1.2f || min < 0f;

                float thr = _confThreshold;
                if (usingSigmoid && thr < 0.5f) thr = 0.5f;

                int fmH = _inH / stride;
                int fmW = _inW / stride;
                int expected = fmH * fmW * _numAnchors;
                if (expected != n)
                    continue;

                for (int idx = 0; idx < n; idx++)
                {
                    float sc = scoresRaw[idx];
                    if (usingSigmoid) sc = Sigmoid(sc);
                    if (sc < thr) continue;

                    int gridIdx = (_numAnchors == 1) ? idx : (idx / _numAnchors);
                    int gy = gridIdx / fmW;
                    int gx = gridIdx - gy * fmW;

                    float cx = gx * stride;
                    float cy = gy * stride;

                    int b0 = idx * 4;
                    float dl = bboxesRaw[b0 + 0] * stride;
                    float dt = bboxesRaw[b0 + 1] * stride;
                    float dr = bboxesRaw[b0 + 2] * stride;
                    float db = bboxesRaw[b0 + 3] * stride;

                    float x1 = (cx - dl) * detScale;
                    float y1 = (cy - dt) * detScale;
                    float x2 = (cx + dr) * detScale;
                    float y2 = (cy + db) * detScale;

                    float x = Math.Max(0, Math.Min(x1, x2));
                    float y = Math.Max(0, Math.Min(y1, y2));
                    float w = Math.Abs(x2 - x1);
                    float h = Math.Abs(y2 - y1);

                    if (w < 8 || h < 8) continue;

                    if (x + w > srcW) w = Math.Max(0, srcW - x);
                    if (y + h > srcH) h = Math.Max(0, srcH - y);
                    if (w <= 0 || h <= 0) continue;

                    dets.Add(new Detection(
                        new Rect((int)Math.Round(x), (int)Math.Round(y),
                                 (int)Math.Round(w), (int)Math.Round(h)),
                        sc));
                }
            }

            if (dets.Count == 0) return dets;
            return Nms(dets, _nmsIouThreshold);
        }

        public Detection? DetectLargest(Mat frameBgr)
        {
            var dets = Detect(frameBgr);
            if (dets.Count == 0) return null;
            return dets.OrderByDescending(d => d.Rect.Width * d.Rect.Height).First();
        }

        private void PrepareScrfdInput(Mat frameBgr, out DenseTensor<float> tensor, out float detScale)
        {
            int ih = frameBgr.Rows;
            int iw = frameBgr.Cols;

            float imgRatio = ih / (float)iw;
            int newW, newH;

            if (imgRatio > _inH / (float)_inW)
            {
                newH = _inH;
                newW = Math.Max(1, (int)Math.Round(newH / imgRatio));
            }
            else
            {
                newW = _inW;
                newH = Math.Max(1, (int)Math.Round(newW * imgRatio));
            }

            detScale = ih / (float)newH;

            using var rgb = new Mat();
            Cv2.CvtColor(frameBgr, rgb, ColorConversionCodes.BGR2RGB);

            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new Size(newW, newH), 0, 0, InterpolationFlags.Linear);

            using var detImg = new Mat(_inH, _inW, MatType.CV_8UC3, Scalar.All(0));
            resized.CopyTo(new Mat(detImg, new Rect(0, 0, newW, newH)));

            tensor = new DenseTensor<float>(new[] { 1, 3, _inH, _inW });

            for (int y = 0; y < _inH; y++)
                for (int x = 0; x < _inW; x++)
                {
                    Vec3b p = detImg.At<Vec3b>(y, x); // RGB
                    tensor[0, 0, y, x] = (p.Item0 - 127.5f) / 128f; // R
                    tensor[0, 1, y, x] = (p.Item1 - 127.5f) / 128f; // G
                    tensor[0, 2, y, x] = (p.Item2 - 127.5f) / 128f; // B
                }
        }

        private static float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));

        private static float IoU(Rect a, Rect b)
        {
            int x1 = Math.Max(a.X, b.X);
            int y1 = Math.Max(a.Y, b.Y);
            int x2 = Math.Min(a.Right, b.Right);
            int y2 = Math.Min(a.Bottom, b.Bottom);

            int iw = Math.Max(0, x2 - x1);
            int ih = Math.Max(0, y2 - y1);
            float inter = iw * ih;
            if (inter <= 0) return 0f;

            float union = (a.Width * a.Height) + (b.Width * b.Height) - inter;
            if (union <= 0) return 0f;

            return inter / union;
        }

        private static List<Detection> Nms(List<Detection> dets, float iouThreshold)
        {
            var sorted = dets.OrderByDescending(d => d.Confidence).ToList();
            var keep = new List<Detection>();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                sorted.RemoveAt(0);
                keep.Add(best);

                sorted = sorted.Where(d => IoU(best.Rect, d.Rect) < iouThreshold).ToList();
            }

            return keep;
        }
    }
}